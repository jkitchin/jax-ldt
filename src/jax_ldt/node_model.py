"""Custom node models.

A node model decides how each node is fitted and how its candidate splits
are scored, in place of the built-in ridge least squares and split kernel.
Pass one to :func:`jax_ldt.tree_core.grow_tree`; the regressor classes
always use the built-in behaviour.

Growth with a node model runs one node at a time; a node model may batch
internally with ``jit``/``vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, eq=False)
class NodeContext:
    """What a node model is given about the training data and the node.

    Fit-wide fields are the same for every node of one fit; ``mask``,
    ``depth`` and ``path`` describe the current node.

    Attributes
    ----------
    X : (N, d)
        Raw training inputs.
    X_t : (N, n_t)
        ``X @ transform_matrix`` (the identity for axis-aligned trees).
    y : (N, T)
        Targets, always 2-D.
    transform_matrix : (d, n_t)
    linear_columns : (n_aug,) bool
        Rows of a node's ``params`` that may be nonzero (bias always true);
        the others correspond to columns outside ``linear_features``.
    mask : (N,) float
        1.0 for rows in this node, 0.0 otherwise, over all N rows.
    depth : int
        Root is 0.
    path : tuple of (feature, threshold, side)
        Splits from the root to this node. ``feature`` indexes ``X_t``;
        side ``"L"`` means ``X_t[:, feature] <= threshold``, ``"R"`` means
        ``>``.
    """

    X: jax.Array
    X_t: jax.Array
    y: jax.Array
    transform_matrix: jax.Array
    linear_columns: np.ndarray
    mask: jax.Array
    depth: int
    path: tuple[tuple[int, float, str], ...]


class NodeModel(Protocol):
    """How nodes are fitted and candidate splits scored.

    ``fit``'s loss and ``score_splits``' scores must be the same quantity:
    growth compares them for ``min_impurity_decrease``.
    """

    def fit(self, ctx: NodeContext) -> tuple[ArrayLike, ArrayLike]:
        """Fit the node on the rows in ``ctx.mask``.

        Returns ``(params, loss)``. ``params`` is ``(n_aug, T)`` float64
        and finite: row 0 is the intercept, row ``i + 1`` the coefficient
        of ``X_t[:, i]``, and rows where ``ctx.linear_columns`` is false
        are exactly 0, so predictions are ``[1, ctx.X_t] @ params``.
        ``loss`` is a finite scalar. A failed fit should raise.
        """
        ...

    def score_splits(
        self,
        ctx: NodeContext,
        thresholds: jax.Array,
        split_features: tuple[int, ...],
        valid: np.ndarray,
    ) -> ArrayLike:
        """Score every candidate split of the node.

        ``thresholds`` is ``(B, K)``: B quantile thresholds per split
        column; ``split_features`` names the K columns of ``X_t``;
        ``valid`` is ``(K, B)``, true where both sides keep at least
        ``min_samples_leaf`` rows.

        Returns ``(K, B)`` scores — transposed relative to
        ``thresholds``: entry ``[k, b]`` scores
        ``X_t[:, split_features[k]] <= thresholds[b, k]`` as the loss of
        the combined prediction over the node's rows, in ``fit``'s units.
        ``inf`` rejects a candidate; entries where ``valid`` is false are
        treated as ``inf``. NaN and ``-inf`` are not allowed.
        """
        ...


def _check_fit_output(
    out: object,
    n_aug: int,
    n_targets: int,
    linear_columns: np.ndarray,
) -> tuple[jax.Array, float]:
    """Validate ``NodeModel.fit`` output; return ``(params, float(loss))``."""
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise ValueError(
            f"node_model.fit must return (params, loss); got {type(out).__name__}."
        )
    params, loss = out
    p = jnp.asarray(params)
    if p.shape != (n_aug, n_targets):
        raise ValueError(
            f"node_model.fit returned params of shape {p.shape}; expected "
            f"(n_aug, n_targets) = {(n_aug, n_targets)}."
        )
    if p.dtype != jnp.float64:
        raise ValueError(f"node_model.fit returned params of dtype {p.dtype}; expected float64.")
    p_np = np.asarray(p)
    if not np.all(np.isfinite(p_np)):
        raise ValueError("node_model.fit returned non-finite params.")
    excluded = np.flatnonzero(~linear_columns)
    bad = excluded[np.any(p_np[excluded] != 0.0, axis=1)]
    if bad.size:
        raise ValueError(
            f"node_model.fit returned nonzero params in rows {bad.tolist()}, which are outside "
            "linear_features and must be exactly 0."
        )
    loss_np = np.asarray(loss)
    if loss_np.shape != ():
        raise ValueError(
            f"node_model.fit returned a loss of shape {loss_np.shape}; expected a scalar."
        )
    if not np.isfinite(loss_np):
        raise ValueError(f"node_model.fit returned a non-finite loss ({float(loss_np)}).")
    return p, float(loss_np)


def _check_scores(scores: ArrayLike, K: int, B: int, valid: np.ndarray) -> np.ndarray:
    """Validate ``NodeModel.score_splits`` output; return a float64 copy, ``inf`` where invalid."""
    s = np.array(scores, dtype=np.float64)
    if s.shape != (K, B):
        raise ValueError(
            f"node_model.score_splits returned shape {s.shape}; expected (K, B) = {(K, B)} "
            "(transposed relative to thresholds)."
        )
    if np.any(np.isnan(s)):
        raise ValueError("node_model.score_splits returned NaN scores; use inf to reject.")
    if np.any(s == -np.inf):
        raise ValueError("node_model.score_splits returned -inf scores; use inf to reject.")
    s[~valid] = np.inf
    return s
