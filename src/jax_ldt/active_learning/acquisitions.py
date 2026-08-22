"""Acquisition functions: score candidate points for the next batch.

Every acquisition implements `.score(model, X_candidates) -> jnp.ndarray`,
returning a 1-D score (higher is better — we always pick max-score).

Acquisitions know about uncertainty calibration; the loop wires them in.

Multi-target support
--------------------
Whether an acquisition can run against a multi-output surrogate is a
property of the *acquisition*, not of the data, so each one advertises it
with the class attribute ``requires_scalar_target``:

* ``True``  — the acquisition is defined against a scalar best-observed-y
  (:class:`ExpectedImprovement`, :class:`ProbabilityOfImprovement`).
* ``False`` — the acquisition only reads per-candidate uncertainty and so
  works for any number of targets (:class:`UncertaintySampler`,
  :class:`MaxVariance`).

:class:`~jax_ldt.active_learning.loop.ActiveLearner` reads the flag and
rejects only the incompatible pairing (scalar-only acquisition + `(n, k)`
targets). Acquisitions that do not define the flag are treated as
scalar-only, which is the conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Protocol, Sequence, Union

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.uncertainty.linprop import linprop_uncertainty


class _ModelLike(Protocol):
    tree_: Optional[Tree]

    def predict(self, X: jnp.ndarray) -> jnp.ndarray: ...


#: Reductions available for collapsing per-target uncertainty to one
#: score per candidate.
_REDUCTIONS = ("max", "mean")


def _check_direction(direction: str) -> None:
    """Reject anything but ``"min"`` / ``"max"``.

    Any other string (``"minimize"``, ``"Min"``, …) previously fell
    through the ``== "min"`` checks and silently flipped the acquisition
    to maximize — a silently inverted optimization direction that only
    surfaces when the campaign converges to the wrong extremum.
    """
    if direction not in ("min", "max"):
        raise ValueError(
            f"direction must be 'min' or 'max'; got {direction!r}."
        )


def _check_reduce(reduce: str) -> None:
    if reduce not in _REDUCTIONS:
        raise ValueError(
            f"reduce must be one of {_REDUCTIONS}; got {reduce!r}."
        )


def _sigma_matrix(model: _ModelLike, X: jnp.ndarray) -> jnp.ndarray:
    """Per-sample, per-target predictive σ as ``(N, T)``.

    ``linprop_uncertainty`` returns ``(N,)`` for single-target trees and
    ``(N, T)`` otherwise; this normalises both to 2-D so callers never
    have to branch on the target count.
    """
    out = jnp.asarray(linprop_uncertainty(model.tree_, X))
    if out.ndim == 1:
        out = out[:, None]
    return out


def _sigma(model: _ModelLike, X: jnp.ndarray) -> jnp.ndarray:
    """Per-sample predictive σ. Always 1-D ``(N,)``.

    ``linprop_uncertainty`` returns ``(N, T)`` for multi-target trees;
    EI / PI score against a scalar best-y, so we reject multi-target
    inputs here rather than letting `(N, T)` flatten into a length-`N·T`
    array that silently misaligns with ``mu``. Exploration acquisitions
    that *can* handle several targets use :func:`_sigma_matrix` plus
    :func:`_reduce_targets` instead.

    Returns a JAX array (kept on-device) so downstream EI/PI scoring
    runs without a numpy round-trip and stays vmap-friendly.
    """
    out = jnp.asarray(linprop_uncertainty(model.tree_, X))
    if out.ndim == 2:
        if out.shape[1] != 1:
            # n_targets is a static Python int on Tree, so this branch
            # decision is JIT-safe.
            raise ValueError(
                f"acquisition functions are single-output, but model.tree_ "
                f"has n_targets={out.shape[1]}. Reduce to a scalar target "
                f"or use a multi-target acquisition (UncertaintySampler, "
                f"MaxVariance)."
            )
        out = out[:, 0]
    return out


def _resolve_target_scales(
    scales: Optional[Union[float, Sequence[float], np.ndarray]], n_targets: int
) -> np.ndarray:
    """Validate/broadcast per-target reference scales to ``(T,)``.

    ``None`` means "no scaling" (all ones). A scalar is broadcast to
    every target. Scales must be finite and strictly positive — a zero or
    negative tolerance has no meaning and would produce inf/NaN scores.
    """
    if scales is None:
        return np.ones((n_targets,), dtype=np.float64)
    s = np.asarray(scales, dtype=np.float64).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, n_targets)
    if s.size != n_targets:
        raise ValueError(
            f"target_scales has {s.size} entries but the model has "
            f"{n_targets} targets."
        )
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            f"target_scales must be finite and strictly positive; got {s.tolist()}."
        )
    return s


def _reduce_targets(
    sigma: jnp.ndarray, scales: np.ndarray, reduce: str
) -> jnp.ndarray:
    """Collapse ``(N, T)`` σ to ``(N,)`` after dividing by ``scales``."""
    z = sigma / jnp.asarray(scales)
    if reduce == "max":
        return jnp.max(z, axis=1)
    return jnp.mean(z, axis=1)


@dataclass
class _ExplorationAcquisition:
    """Shared multi-target plumbing for uncertainty-only acquisitions.

    Parameters
    ----------
    target_scales :
        Per-target reference scale used to make targets comparable before
        they are reduced to a single score. The natural choice is the
        *tolerance actually required* on each target: a batch is then
        picked for whichever target is worst **relative to the accuracy
        needed on it**, rather than worst in absolute units. A scalar is
        broadcast across targets.

        If left at ``None``, :class:`~jax_ldt.active_learning.loop.ActiveLearner`
        fills in a fallback each round: the observed per-target standard
        deviation of `y` (see :meth:`update_target_scales`). Used
        standalone with no scales at all, targets are reduced in **raw
        units**, so a large-magnitude target can dominate the batch.
    reduce :
        ``"max"`` (default) or ``"mean"`` over the scaled per-target σ.
        ``"max"`` targets the currently worst-served output; ``"mean"``
        spreads effort across all of them.

    Single-target models bypass all of this: the score is the raw σ (or
    σ²), unchanged and unscaled, exactly as before multi-target support.
    """

    requires_scalar_target: ClassVar[bool] = False

    target_scales: Optional[Union[float, Sequence[float], np.ndarray]] = None
    reduce: str = "max"
    # Fallback scales derived from observed y; only consulted when the
    # user did not pass `target_scales` explicitly.
    _observed_scales: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _check_reduce(self.reduce)

    def update_target_scales(self, y_observed: np.ndarray) -> None:
        """Record fallback per-target scales from observed spread.

        Called by :class:`~jax_ldt.active_learning.loop.ActiveLearner`
        before scoring. The fallback is each target's standard deviation
        over the observed data; degenerate (zero / non-finite) spreads
        fall back to 1.0. An explicit ``target_scales`` always wins.
        """
        y = np.asarray(y_observed, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[0] < 2:
            self._observed_scales = None
            return
        s = np.std(y, axis=0)
        self._observed_scales = np.where(np.isfinite(s) & (s > 0.0), s, 1.0)

    def _reduced_sigma(self, model: _ModelLike, X: jnp.ndarray) -> jnp.ndarray:
        sigma = _sigma_matrix(model, X)
        n_targets = sigma.shape[1]
        if n_targets == 1:
            return sigma[:, 0]
        scales = (
            self.target_scales
            if self.target_scales is not None
            else self._observed_scales
        )
        return _reduce_targets(
            sigma, _resolve_target_scales(scales, n_targets), self.reduce
        )


@dataclass
class UncertaintySampler(_ExplorationAcquisition):
    """Maximum-uncertainty acquisition.

    Score = σ_pred(x). Pure exploration. Multi-target models are scored
    by reducing the per-target σ with the scaling described in
    :class:`_ExplorationAcquisition` (default: per-candidate max of
    σ_t / scale_t), so the score is dimensionless when T > 1 and in σ
    units when T == 1.
    """

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        return np.asarray(self._reduced_sigma(model, X_candidates))


@dataclass
class MaxVariance(_ExplorationAcquisition):
    """Maximum predictive-variance acquisition.

    Score = σ_pred(x)². Argmax-equivalent to :class:`UncertaintySampler`
    (squaring is monotone on σ ≥ 0), but the *score values* are σ² —
    matching the class name. Use this when downstream code consumes the
    raw score (e.g., portfolio acquisitions or adaptive temperature
    schedules) and expects variance units.

    For multi-target models the reduction happens on σ (see
    :class:`_ExplorationAcquisition`) and the reduced value is then
    squared, which keeps the argmax-equivalence with
    :class:`UncertaintySampler` under both reductions.
    """

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        sigma = self._reduced_sigma(model, X_candidates)
        return np.asarray(sigma * sigma)


def _scalar_best(y_observed: np.ndarray, direction: str) -> Optional[float]:
    """Best observed y for a scalar objective, or None if there is none.

    Rejects multi-target `y` rather than flattening it: `min` over a
    ravelled `(n, k)` array is the min over *all* targets at once, which
    is not the best-y of any single objective.
    """
    y = np.asarray(y_observed, dtype=np.float64)
    if y.ndim > 1 and y.shape[-1] > 1:
        raise ValueError(
            f"expected a single-target objective, got y with shape {y.shape}. "
            "EI / PI score against a scalar best-y; use UncertaintySampler or "
            "MaxVariance for multi-target exploration, or pass a scalarisation."
        )
    y = y.reshape(-1)
    if y.size == 0:
        return None
    return float(np.min(y) if direction == "min" else np.max(y))


@dataclass
class ExpectedImprovement:
    """Gaussian expected improvement.

    Treats σ_pred as a Gaussian std around the mean prediction. Score is
    EI relative to the best observed `y` so far.

    Single-target only (``requires_scalar_target = True``): "the best y
    so far" is not defined for a vector objective.

    direction: "min" → minimize y; "max" → maximize y. Default min.
    xi: exploration parameter (added to the gap). Default 0.0.
    """

    requires_scalar_target: ClassVar[bool] = True

    direction: str = "min"
    xi: float = 0.0
    _y_best: Optional[float] = None

    def __post_init__(self) -> None:
        _check_direction(self.direction)

    def update_best(self, y_observed: np.ndarray) -> None:
        best = _scalar_best(y_observed, self.direction)
        if best is not None:
            self._y_best = best

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        if self._y_best is None:
            raise RuntimeError("Call update_best(y) before scoring.")
        mu = jnp.asarray(model.predict(X_candidates)).reshape(-1)
        sigma = jnp.maximum(_sigma(model, X_candidates).reshape(-1), 1e-9)

        if self.direction == "min":
            improvement = self._y_best - mu - self.xi
        else:
            improvement = mu - self._y_best - self.xi
        z = improvement / sigma
        ei = improvement * jstats.norm.cdf(z) + sigma * jstats.norm.pdf(z)
        return np.asarray(ei)


@dataclass
class ProbabilityOfImprovement:
    """Gaussian probability of improvement against best-observed.

    Single-target only (``requires_scalar_target = True``).
    """

    requires_scalar_target: ClassVar[bool] = True

    direction: str = "min"
    xi: float = 0.0
    _y_best: Optional[float] = None

    def __post_init__(self) -> None:
        _check_direction(self.direction)

    def update_best(self, y_observed: np.ndarray) -> None:
        best = _scalar_best(y_observed, self.direction)
        if best is not None:
            self._y_best = best

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        if self._y_best is None:
            raise RuntimeError("Call update_best(y) before scoring.")
        mu = jnp.asarray(model.predict(X_candidates)).reshape(-1)
        sigma = jnp.maximum(_sigma(model, X_candidates).reshape(-1), 1e-9)
        if self.direction == "min":
            z = (self._y_best - mu - self.xi) / sigma
        else:
            z = (mu - self._y_best - self.xi) / sigma
        return np.asarray(jstats.norm.cdf(z))
