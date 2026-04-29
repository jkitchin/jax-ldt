"""Linear-propagation uncertainty.

For a leaf with `n` training points, sample mean `μ`, sample variance
`σ²`, and leaf MSE `m`:

    σ_pred(x) = sqrt(m/n + Σⱼ (xⱼ - μⱼ)² / ((n-1) σⱼ²))

This is a port of upstream's `TorchLinearRegression.linprop_uncertainty`,
adapted to operate on the per-leaf UQ blob already populated during
tree growth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_ldt._types import Tree
from jax_ldt.tree_core import _route_one


def _resolve_tree(model_or_tree) -> Tree:
    """Accept either a Tree pytree or a fitted regressor (with .tree_)."""
    if isinstance(model_or_tree, Tree):
        return model_or_tree
    tree = getattr(model_or_tree, "tree_", None)
    if tree is None:
        raise TypeError(
            "Pass either a Tree pytree or a fitted regressor (with `.tree_`)."
        )
    return tree


def _per_sample_uncertainty(tree: Tree, x_in: jnp.ndarray) -> jnp.ndarray:
    """Compute σ_pred for one sample (shape: (n_features_in,)).

    Edge cases:

    * Effective ``n < 2``: the deviation term is undefined (no
      within-leaf variance to scale against). We return only the base
      ``sqrt(mse / n)`` term in that case rather than fabricating a
      pseudo-denominator of ``1``.
    * Constant feature inside a leaf (``x_var <= eps``): the leaf's
      linear model cannot have used that feature (no within-leaf
      variation), so we drop it from the deviation sum entirely. The
      old code clamped ``x_var`` to ``1e-12`` which would inflate σ by
      ~1e13 for any test point a unit away from the leaf mean.

    The ``eps = 1e-10`` floor matches the JAX 64-bit zero threshold we
    use elsewhere (see ``_validation.py``); features with variance below
    this are treated as constant within the leaf.
    """
    # Floor for "effectively zero" variance under JAX float64 arithmetic.
    eps = 1e-10

    x_t = x_in @ tree.transform_matrix  # (n_features_transformed,)
    leaf_id = _route_one(tree, x_t)

    if tree.leaf_uq is None:
        raise RuntimeError("Tree has no leaf UQ; refit model.")

    uq = tree.leaf_uq
    n = uq.n[leaf_id].astype(jnp.float64)
    x_mean = uq.x_mean[leaf_id]  # (n_transformed,)
    x_var = uq.x_var[leaf_id]
    mse = uq.mse[leaf_id]  # (n_targets,)

    safe_n = jnp.where(n > 0.0, n, 1.0)
    base = mse / safe_n  # (n_targets,)

    # Per-feature deviation contribution: drop constant features (the
    # leaf model could not have used them, so they should not inflate σ).
    feature_active = x_var > eps
    safe_var = jnp.where(feature_active, x_var, 1.0)
    per_feature = jnp.where(
        feature_active,
        (x_t - x_mean) ** 2 / safe_var,
        0.0,
    )
    raw_deviation = jnp.sum(per_feature)

    # When n < 2, the deviation term is undefined; vanish it. Otherwise
    # divide by (n-1). Use a safe denominator inside the where to avoid
    # NaN propagation from the discarded branch.
    has_variance = n >= 2.0
    safe_n_minus_one = jnp.where(has_variance, n - 1.0, 1.0)
    deviation = jnp.where(has_variance, raw_deviation / safe_n_minus_one, 0.0)

    sigma_sq = base + deviation  # broadcast: (n_targets,)
    return jnp.sqrt(jnp.clip(sigma_sq, min=0.0))


def linprop_uncertainty(model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
    """Per-sample uncertainty (n_samples, n_targets) or (n_samples,) if T=1.

    Accepts either a fitted regressor (with `.tree_`) or a `Tree` pytree.
    """
    tree = _resolve_tree(model_or_tree)
    X = jnp.asarray(X, dtype=jnp.float64)
    out = jax.vmap(lambda x: _per_sample_uncertainty(tree, x))(X)  # (N, T)
    if tree.n_targets == 1:
        out = out[:, 0]
    return out


class LinearPropagationUQ:
    """Marker class that selects linear-propagation UQ.

    UQ params are always saved during tree growth (cheap enough), so
    this class just signals the desired predict path. Accepts either a
    fitted regressor or a `Tree` pytree.
    """

    def predict(self, model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
        return linprop_uncertainty(model_or_tree, X)
