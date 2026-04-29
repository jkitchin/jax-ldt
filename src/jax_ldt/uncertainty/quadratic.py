"""Quadratic-difference uncertainty.

After fitting a linear leaf model, also fit a quadratic-feature variant
on the same leaf data. The absolute difference between the two
predictions is reported as a "model-form sensitivity" proxy.

Implementation note: rather than refit at predict time, we pre-compute
the quadratic params per leaf at calibration time and store them in
a parallel array. Memory cost: O(n_leaves * n_aug_quadratic^2 * T),
which for typical sizes (n_aug ≈ 10–50, T = 1) is small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.linear_regression import add_bias
from jax_ldt.tree_core import _fit_many_sides, _route_one
from jax_ldt.uncertainty.linprop import _resolve_tree


def _quadratic_features(X_aug: jnp.ndarray) -> jnp.ndarray:
    """Append cross-products X_i * X_j for i <= j to the augmented features.

    X_aug : (N, n_aug)  -- already has the bias column at index 0
    Returns (N, n_aug + n_aug*(n_aug+1)/2 - 1) with the all-ones column kept once.
    """
    N, F = X_aug.shape
    # outer products: (N, F, F)
    outer = X_aug[:, :, None] * X_aug[:, None, :]
    # take the upper triangle (including diagonal). Skip [0,0] since it's just 1.
    iu, ju = jnp.triu_indices(F)
    cross = outer[:, iu, ju]  # (N, F*(F+1)/2)
    return cross  # already includes the bias-squared term at index 0 (== 1)


@dataclass
class QuadraticUQState:
    """Pre-computed quadratic params per leaf."""

    leaf_quad_params: jnp.ndarray  # (n_nodes, n_quad_features, n_targets)


def calibrate_quadratic(
    model_or_tree,
    X: jnp.ndarray,
    y: jnp.ndarray,
    ridge: float = 1e-5,
) -> QuadraticUQState:
    """Refit each leaf with quadratic features. Returns state usable by predict.

    Accepts either a fitted regressor or a `Tree` pytree.
    """
    tree = _resolve_tree(model_or_tree)
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if y.ndim == 1:
        y = y[:, None]

    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)
    X_quad = _quadratic_features(X_aug)
    n_quad = X_quad.shape[1]
    T = y.shape[1]

    # Route each sample to its leaf, then materialise once on host so
    # the per-leaf mask construction below doesn't host-sync per node.
    leaf_ids = np.asarray(jax.vmap(lambda x: _route_one(tree, x))(X_t))
    n_nodes = int(tree.n_nodes)

    # Build a (n_nodes, N) leaf-membership matrix on host. Non-leaf
    # nodes get an all-zero row, which the ridge regulariser absorbs to
    # produce a zero-coefficient fit (overwritten below to keep the
    # zero-init invariant that downstream code relies on).
    is_leaf_np = np.asarray(tree.is_leaf)
    leaf_node_ids = np.where(is_leaf_np)[0]
    masks_np = np.zeros((n_nodes, X_quad.shape[0]), dtype=np.float64)
    if leaf_node_ids.size > 0:
        membership = (leaf_ids[None, :] == leaf_node_ids[:, None]).astype(np.float64)
        masks_np[leaf_node_ids] = membership
    masks = jnp.asarray(masks_np)

    # One vmapped JIT call instead of `n_nodes` Python-loop fits with
    # per-iteration host syncs. Non-leaf rows fit to zero coefficients
    # under the ridge prior; we explicitly zero them again to preserve
    # the public contract that non-leaf nodes have zero quad params.
    fitted = _fit_many_sides(X_quad, y, masks, ridge)  # (n_nodes, n_quad, T)
    keep = jnp.asarray(is_leaf_np[:, None, None].astype(np.float64))
    quad_params = fitted * keep

    return QuadraticUQState(leaf_quad_params=quad_params)


def quadratic_uncertainty(
    model_or_tree, state: QuadraticUQState, X: jnp.ndarray
) -> jnp.ndarray:
    """|linear_pred - quadratic_pred| per sample.

    Accepts either a fitted regressor or a `Tree` pytree.
    """
    tree = _resolve_tree(model_or_tree)
    X = jnp.asarray(X, dtype=jnp.float64)
    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)
    X_quad = _quadratic_features(X_aug)

    leaf_ids = jax.vmap(lambda x: _route_one(tree, x))(X_t)

    # Linear prediction
    lin_params = tree.leaf_params[leaf_ids]  # (N, n_aug, T)
    yh_lin = jnp.einsum("nf,nft->nt", X_aug, lin_params)

    # Quadratic prediction
    quad_params_per_sample = state.leaf_quad_params[leaf_ids]  # (N, n_quad, T)
    yh_quad = jnp.einsum("nq,nqt->nt", X_quad, quad_params_per_sample)

    out = jnp.abs(yh_lin - yh_quad)
    if tree.n_targets == 1:
        out = out[:, 0]
    return out


class QuadraticUQ:
    """Marker class for quadratic-difference UQ. Stateful: holds calibration."""

    def __init__(self, ridge: float = 1e-5) -> None:
        self.ridge = ridge
        self.state_: Optional[QuadraticUQState] = None
        self._tree: Optional[Tree] = None

    def calibrate(self, model_or_tree, X: jnp.ndarray, y: jnp.ndarray) -> "QuadraticUQ":
        self._tree = _resolve_tree(model_or_tree)
        self.state_ = calibrate_quadratic(self._tree, X, y, ridge=self.ridge)
        return self

    def predict(self, model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
        if self.state_ is None or self._tree is None:
            raise RuntimeError("Call .calibrate(model, X, y) before .predict.")
        # Allow caller to pass either form; we route through the tree
        # captured at calibration time.
        _ = _resolve_tree(model_or_tree)
        return quadratic_uncertainty(self._tree, self.state_, X)
