"""Frozen JAX-pytree dataclasses for the trained tree representation.

The `Tree` dataclass uses parallel arrays (one entry per node) to encode
the trained tree topology + per-leaf linear models. This layout is
JIT/vmap-friendly and matches `discopt.nn.tree.DecisionTree`, so the
discopt adapter is one cheap copy.

Per-node convention:
- non-leaf: `is_leaf=False`, `feature` and `threshold` set, `left/right`
  point to child node ids; `leaf_params` row is unused (zero-padded).
- leaf:     `is_leaf=True`,  `feature=-1`, `threshold=NaN`, `left=right=-1`,
  `leaf_params` row holds the (n_features+1, n_targets) augmented
  regression coefficients [intercept; weights].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LeafUQ:
    """Per-leaf linear-propagation uncertainty parameters.

    Each array is (n_nodes, ...); rows for non-leaf nodes are zero-padded.
    """

    n: jax.Array  # (n_nodes,) int — sample count in the leaf
    x_mean: jax.Array  # (n_nodes, n_features) float — feature means
    x_var: jax.Array  # (n_nodes, n_features) float — feature variances
    mse: jax.Array  # (n_nodes, n_targets) float — leaf MSE


def _leaf_uq_flatten(u: LeafUQ) -> tuple[tuple[jax.Array, ...], None]:
    return ((u.n, u.x_mean, u.x_var, u.mse), None)


def _leaf_uq_unflatten(_aux: None, children: tuple[jax.Array, ...]) -> LeafUQ:
    n, x_mean, x_var, mse = children
    return LeafUQ(n=n, x_mean=x_mean, x_var=x_var, mse=mse)


jax.tree_util.register_pytree_node(LeafUQ, _leaf_uq_flatten, _leaf_uq_unflatten)


@dataclass(frozen=True)
class Tree:
    """Trained tree: parallel-array topology + per-leaf linear models.

    Dynamic (pytree) fields are jax.Arrays. Static (auxiliary) fields are
    Python scalars / tuples; they participate in `==` comparisons but not
    in tree-map traversals.
    """

    # ---- dynamic (children of the pytree) ----
    is_leaf: jax.Array  # (n_nodes,) bool
    feature: jax.Array  # (n_nodes,) int32 — split feature in transformed space; -1 if leaf
    threshold: jax.Array  # (n_nodes,) float — split threshold; NaN if leaf
    left: jax.Array  # (n_nodes,) int32 — left child id; -1 if leaf
    right: jax.Array  # (n_nodes,) int32 — right child id; -1 if leaf
    leaf_params: jax.Array  # (n_nodes, n_aug, n_targets) — [bias; weights] for leaves
    transform_matrix: jax.Array  # (n_features_in, n_features_transformed)
    leaf_uq: Optional[LeafUQ]

    # ---- static (auxiliary) ----
    n_features_in: int = field(metadata={"static": True})
    n_features_transformed: int = field(metadata={"static": True})
    n_targets: int = field(metadata={"static": True})
    linear_features: tuple[int, ...] = field(metadata={"static": True})
    categorical_features: tuple[int, ...] = field(metadata={"static": True})

    # ---- convenience ----
    @property
    def n_nodes(self) -> int:
        return int(self.is_leaf.shape[0])

    @property
    def n_leaves(self) -> int:
        return int(jnp.sum(self.is_leaf))

    def replace(self, **changes: Any) -> "Tree":
        return replace(self, **changes)


# Pytree wire types: typed tuples so the unflatten body can pass values
# straight to the dataclass without `cast` or `# type: ignore`.
_TreeChildren = tuple[
    jax.Array,  # is_leaf
    jax.Array,  # feature
    jax.Array,  # threshold
    jax.Array,  # left
    jax.Array,  # right
    jax.Array,  # leaf_params
    jax.Array,  # transform_matrix
    Optional[LeafUQ],
]
_TreeAux = tuple[int, int, int, tuple[int, ...], tuple[int, ...]]


def _tree_flatten(t: Tree) -> tuple[_TreeChildren, _TreeAux]:
    children: _TreeChildren = (
        t.is_leaf,
        t.feature,
        t.threshold,
        t.left,
        t.right,
        t.leaf_params,
        t.transform_matrix,
        t.leaf_uq,
    )
    aux: _TreeAux = (
        t.n_features_in,
        t.n_features_transformed,
        t.n_targets,
        t.linear_features,
        t.categorical_features,
    )
    return children, aux


def _tree_unflatten(aux: _TreeAux, children: _TreeChildren) -> Tree:
    is_leaf, feature, threshold, left, right, leaf_params, transform_matrix, leaf_uq = children
    n_features_in, n_features_transformed, n_targets, linear_features, categorical_features = aux
    return Tree(
        is_leaf=is_leaf,
        feature=feature,
        threshold=threshold,
        left=left,
        right=right,
        leaf_params=leaf_params,
        transform_matrix=transform_matrix,
        leaf_uq=leaf_uq,
        n_features_in=n_features_in,
        n_features_transformed=n_features_transformed,
        n_targets=n_targets,
        linear_features=linear_features,
        categorical_features=categorical_features,
    )


jax.tree_util.register_pytree_node(Tree, _tree_flatten, _tree_unflatten)
