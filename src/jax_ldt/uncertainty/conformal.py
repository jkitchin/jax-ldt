"""Mondrian leaf conformal prediction.

After fitting a tree, we route a calibration set through and aggregate
absolute residuals per leaf. The (1 - α) empirical quantile of those
residuals is used as the prediction-interval half-width for any test
point routed to that leaf.

Sparse-leaf fallback strategies (configurable):

- ``"global"`` (default): leaves with fewer than ``min_calibration_per_leaf``
  points use the global residual quantile across **all** calibration
  data. This keeps every test point covered, but sacrifices the strict
  Mondrian exchangeability guarantee for those leaves: the per-leaf
  coverage on sparse leaves can deviate from ``1 - α`` because the
  global quantile is a mixture across leaves with different residual
  distributions. Marginal coverage across all test points remains close
  to nominal in practice; per-leaf coverage does not.

- ``"skip"``: sparse leaves are flagged. Calling ``predict_interval``
  for points routed to a sparse leaf returns ``(NaN, NaN)`` instead of
  silently relying on a leaky guarantee. Use this when valid per-leaf
  coverage is critical (e.g., safety-critical applications).

In both cases a single ``UserWarning`` enumerates the affected leaves
when calibration runs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.tree_core import _route_one, predict


class _FittedTreeLike(Protocol):
    tree_: Optional[Tree]
    n_targets_: Optional[int]

    def predict(self, X: jnp.ndarray) -> jnp.ndarray: ...


@dataclass
class ConformalState:
    leaf_halfwidth: jnp.ndarray  # (n_nodes, n_targets)
    global_halfwidth: jnp.ndarray  # (n_targets,) — fallback


class ConformalCalibrator:
    """Mondrian leaf conformal calibrator.

    Parameters
    ----------
    alpha : float
        Miscoverage level. ``1 - alpha`` is the target marginal coverage
        (e.g. ``alpha=0.1`` → 90% intervals).
    mondrian : bool
        If True (default), aggregate residuals per leaf. If False, use a
        single global quantile and ignore ``sparse_leaf_strategy``.
    min_calibration_per_leaf : int
        Leaves with fewer calibration points trigger the sparse-leaf
        fallback (see ``sparse_leaf_strategy``).
    sparse_leaf_strategy : {"global", "skip"}, default "global"
        How to treat leaves with fewer calibration points than
        ``min_calibration_per_leaf``. ``"global"`` borrows the global
        residual quantile (preserves marginal coverage but loses
        per-leaf exchangeability); ``"skip"`` returns ``NaN`` intervals
        at predict time so the user can detect and handle them.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        mondrian: bool = True,
        min_calibration_per_leaf: int = 5,
        sparse_leaf_strategy: str = "global",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if sparse_leaf_strategy not in {"global", "skip"}:
            raise ValueError(
                f"sparse_leaf_strategy must be 'global' or 'skip'; got {sparse_leaf_strategy!r}"
            )
        self.alpha = float(alpha)
        self.mondrian = bool(mondrian)
        self.min_calibration_per_leaf = int(min_calibration_per_leaf)
        self.sparse_leaf_strategy = sparse_leaf_strategy
        self.state_: Optional[ConformalState] = None
        self._tree: Optional[Tree] = None
        self._sparse_leaves: tuple[int, ...] = ()

    def calibrate(self, model: _FittedTreeLike, X_cal: jnp.ndarray, y_cal: jnp.ndarray) -> "ConformalCalibrator":
        if model.tree_ is None:
            raise RuntimeError("Model must be fitted before calibration.")
        tree = model.tree_
        X_cal = jnp.asarray(X_cal)
        y_cal = jnp.asarray(y_cal)
        if y_cal.ndim == 1:
            y_cal = y_cal[:, None]

        # `model.predict` returns (N,) for single-target trees and (N, T)
        # for multi-target. Normalise to (N, T) without a fragile
        # transpose: shape coercion via `atleast_2d` would silently treat
        # a (1, T) honest prediction as a (T, 1) column and flip it.
        yh = jnp.asarray(model.predict(X_cal))
        if yh.ndim == 1:
            yh = yh[:, None]
        elif yh.ndim != 2:
            raise ValueError(
                f"model.predict(X_cal) returned an array with ndim={yh.ndim}; "
                "expected 1-D or 2-D."
            )
        if yh.shape[0] != y_cal.shape[0]:
            raise ValueError(
                f"model.predict(X_cal) returned shape {tuple(yh.shape)} but "
                f"y_cal has {y_cal.shape[0]} rows."
            )

        residuals = jnp.abs(y_cal - yh)  # (N_cal, T)

        # Quantile correction: empirical (1-α) on N points needs the
        # ceil((N+1)(1-α)) / N quantile for valid finite-sample coverage.
        # When (N+1)(1-α) > N the quantile is clipped to 1.0 (max
        # residual), and the strict finite-sample CP guarantee no longer
        # holds — warn so the caller knows their calibration set is too
        # small for the requested α.
        N = residuals.shape[0]
        raw_q_level = np.ceil((N + 1) * (1.0 - self.alpha)) / N
        if raw_q_level > 1.0:
            warnings.warn(
                f"Calibration set size N={N} too small for alpha={self.alpha}: "
                f"finite-sample conformal coverage requires N >= "
                f"ceil(1/alpha) - 1 = {int(np.ceil(1.0 / self.alpha)) - 1}. "
                f"Using max-residual halfwidth as a fallback; the strict "
                f"(1 - alpha) coverage guarantee no longer holds.",
                UserWarning,
                stacklevel=2,
            )
        q_level = min(1.0, raw_q_level)

        # Global fallback
        global_hw = jnp.quantile(residuals, q_level, axis=0)  # (T,)

        if not self.mondrian:
            leaf_hw = jnp.broadcast_to(global_hw, (tree.n_nodes, residuals.shape[1]))
        else:
            X_t_cal = X_cal @ tree.transform_matrix
            leaf_ids_cal = np.asarray(jax.vmap(lambda x: _route_one(tree, x))(X_t_cal))
            residuals_np = np.asarray(residuals)
            n_nodes = tree.n_nodes
            T = residuals_np.shape[1]
            global_np = np.asarray(global_hw)

            leaf_hw_np = np.broadcast_to(global_np, (n_nodes, T)).copy()
            sparse_leaves: list[int] = []

            for nid in range(n_nodes):
                if not bool(tree.is_leaf[nid]):
                    continue
                idx = np.where(leaf_ids_cal == nid)[0]
                if len(idx) < self.min_calibration_per_leaf:
                    sparse_leaves.append(nid)
                    if self.sparse_leaf_strategy == "skip":
                        # Mark this leaf as un-calibrated; predict_interval
                        # returns NaN for points routed here.
                        leaf_hw_np[nid] = np.nan
                    # else: keep the global broadcast as the fallback.
                    continue
                leaf_residuals = residuals_np[idx]  # (n_leaf_cal, T)
                leaf_q_level = min(1.0, np.ceil((len(idx) + 1) * (1.0 - self.alpha)) / len(idx))
                leaf_hw_np[nid] = np.quantile(leaf_residuals, leaf_q_level, axis=0)

            if sparse_leaves:
                preview = sparse_leaves[:10]
                more = "" if len(sparse_leaves) <= 10 else f" (+{len(sparse_leaves) - 10} more)"
                action = (
                    "fell back to the global residual quantile (loses per-leaf "
                    "exchangeability; marginal coverage is preserved)"
                    if self.sparse_leaf_strategy == "global"
                    else "are flagged; predict_interval returns NaN for points routed there"
                )
                warnings.warn(
                    f"{len(sparse_leaves)} leaves had fewer than "
                    f"{self.min_calibration_per_leaf} calibration points and "
                    f"{action} (leaf ids: {preview}{more})",
                    UserWarning,
                    stacklevel=2,
                )

            self._sparse_leaves = tuple(sparse_leaves)
            leaf_hw = jnp.asarray(leaf_hw_np)

        self.state_ = ConformalState(leaf_halfwidth=leaf_hw, global_halfwidth=global_hw)
        self._tree = tree
        return self

    def predict_interval(
        self, X: jnp.ndarray, model: Optional[_FittedTreeLike] = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (lower, upper) prediction intervals.

        If `model` is provided, use it to recompute point predictions;
        otherwise, the calibrator must have been calibrated against a
        model whose `tree_` matches `self._tree`.
        """
        if self.state_ is None or self._tree is None:
            raise RuntimeError("Call calibrate(...) first.")
        tree = self._tree
        X = jnp.asarray(X)

        if model is not None:
            yh = jnp.asarray(model.predict(X))
        else:
            yh = predict(tree, X)
        if yh.ndim == 1:
            yh = yh[:, None]

        X_t = X @ tree.transform_matrix
        leaf_ids = jax.vmap(lambda x: _route_one(tree, x))(X_t)
        hw = self.state_.leaf_halfwidth[leaf_ids]  # (N, T)
        lo = yh - hw
        hi = yh + hw
        if tree.n_targets == 1:
            lo = lo[:, 0]
            hi = hi[:, 0]
        return lo, hi


def fit_with_conformal(
    model,
    X,
    y,
    *,
    calibration_size: float = 0.2,
    alpha: float = 0.1,
    mondrian: bool = True,
    rng_seed: int = 0,
) -> tuple[object, ConformalCalibrator]:
    """Convenience: fit `model` on a held-in split, calibrate on the held-out remainder.

    Returns (fitted_model, calibrator).
    """
    if not 0.0 < calibration_size < 1.0:
        raise ValueError("calibration_size must be in (0, 1)")
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(rng_seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_cal = int(np.ceil(calibration_size * n))
    cal_idx = perm[:n_cal]
    fit_idx = perm[n_cal:]
    model.fit(X[fit_idx], y[fit_idx])
    calib = ConformalCalibrator(alpha=alpha, mondrian=mondrian).calibrate(
        model, X[cal_idx], y[cal_idx]
    )
    return model, calib
