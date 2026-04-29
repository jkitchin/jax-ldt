"""G-5: min_impurity_decrease pruning semantics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt import LinearTreeRegressor


def test_zero_min_impurity_grows_tree(rng) -> None:
    """Default 0.0: any improvement is accepted, tree should split."""
    X = rng.uniform(-2, 2, size=(200, 1))
    y = np.sign(X[:, 0]) + 0.05 * rng.standard_normal(200)
    model = LinearTreeRegressor(
        max_depth=4, max_bins=10, min_samples_leaf=10, min_impurity_decrease=0.0
    ).fit(X, y)
    assert model.num_leaves >= 2, "tree should have split with default min_impurity_decrease"


def test_huge_min_impurity_prunes_to_root(rng) -> None:
    """A very large threshold forces the tree to stay a single leaf."""
    X = rng.uniform(-2, 2, size=(200, 1))
    y = np.sign(X[:, 0]) + 0.05 * rng.standard_normal(200)
    model = LinearTreeRegressor(
        max_depth=4,
        max_bins=10,
        min_samples_leaf=10,
        min_impurity_decrease=1e6,
    ).fit(X, y)
    assert model.num_leaves == 1, (
        f"tree should have stayed at root with huge min_impurity_decrease; "
        f"got {model.num_leaves} leaves"
    )


def test_intermediate_threshold_prunes_some_splits(rng) -> None:
    """A small but positive threshold should produce fewer leaves than 0.0."""
    X = rng.uniform(-2, 2, size=(300, 2))
    y = X[:, 0] ** 2 + 0.05 * rng.standard_normal(300)

    permissive = LinearTreeRegressor(
        max_depth=5, max_bins=8, min_samples_leaf=10, min_impurity_decrease=0.0
    ).fit(X, y)
    strict = LinearTreeRegressor(
        max_depth=5, max_bins=8, min_samples_leaf=10, min_impurity_decrease=0.5
    ).fit(X, y)
    assert strict.num_leaves <= permissive.num_leaves, (
        f"tighter pruning should produce ≤ leaves: strict={strict.num_leaves}, "
        f"permissive={permissive.num_leaves}"
    )


def test_zero_decrease_threshold_is_default() -> None:
    """Default value is 0.0 (was previously -inf)."""
    model = LinearTreeRegressor()
    assert model.min_impurity_decrease == 0.0
