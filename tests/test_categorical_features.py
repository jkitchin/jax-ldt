"""Coverage for documented hyperparameters that the rest of the suite
underexercises:

  * ``categorical_features`` at the regressor level — previously only
    tested on the bare hyperplane builder, never via the public LMDT/HT
    surface.
  * ``min_samples_leaf`` as a float fraction of N — public docstring
    advertises the float path; it had no test.

Both gaps are pinned in GAPS.md G-22.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor


def test_lmdt_categorical_feature_is_never_used_for_splits(rng) -> None:
    """A column declared categorical must never appear as a split feature."""
    n = 200
    cat = rng.integers(0, 3, size=n).astype(np.float64)
    cont = rng.uniform(-2, 2, size=n)
    X = np.column_stack([cat, cont]).astype(np.float64)
    # Target depends on the continuous column only — splits should land there.
    y = (cont ** 2 + 0.05 * rng.standard_normal(n)).astype(np.float64)

    model = LinearTreeRegressor(
        max_depth=3,
        max_bins=5,
        min_samples_leaf=15,
        categorical_features=(0,),
    ).fit(X, y)

    feature = np.asarray(model.tree_.feature)
    is_leaf = np.asarray(model.tree_.is_leaf)
    used_for_splits = set(int(f) for f in feature[~is_leaf])
    assert 0 not in used_for_splits, (
        f"categorical column 0 was used for a split: {used_for_splits}"
    )


def test_ht_categorical_feature_is_excluded_from_lifted_directions(rng) -> None:
    """Categorical declarations propagate into the lifted hyperplane block."""
    n = 200
    cat = rng.integers(0, 4, size=n).astype(np.float64)
    cont = rng.uniform(-1, 1, size=(n, 2))
    X = np.column_stack([cat, cont]).astype(np.float64)
    y = (cont[:, 0] + cont[:, 1] + 0.05 * rng.standard_normal(n)).astype(np.float64)

    no_cat = HyperplaneTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=15, max_weight=1, num_terms=2
    ).fit(X, y)
    with_cat = HyperplaneTreeRegressor(
        max_depth=3,
        max_bins=5,
        min_samples_leaf=15,
        max_weight=1,
        num_terms=2,
        categorical_features=(0,),
    ).fit(X, y)

    A_no = np.asarray(no_cat.tree_.transform_matrix)
    A_yes = np.asarray(with_cat.tree_.transform_matrix)
    # Excluding column 0 from oblique permutations gives strictly fewer
    # (or equal) lifted columns.
    assert A_yes.shape[1] <= A_no.shape[1]
    # Beyond the identity block, the categorical column must have a zero
    # coefficient in every lifted-only column.
    n_in = X.shape[1]
    if A_yes.shape[1] > n_in:
        np.testing.assert_array_equal(A_yes[0, n_in:], 0.0)


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25])
def test_min_samples_leaf_as_float_fraction(rng, frac) -> None:
    """A float in (0, 1) is interpreted as a fraction of N.

    Every produced leaf must hold at least ceil(frac * N) samples (the
    `grow_tree` resolution rule, with the documented floor of 3).
    """
    n = 240
    X = rng.uniform(-2, 2, size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1]) + 0.05 * rng.standard_normal(n)

    model = LinearTreeRegressor(
        max_depth=4, max_bins=5, min_samples_leaf=frac
    ).fit(X, y)

    expected_floor = max(3, int(np.ceil(frac * n)))
    leaf_n = np.asarray(model.tree_.leaf_uq.n)
    is_leaf = np.asarray(model.tree_.is_leaf)
    actual = leaf_n[is_leaf]
    assert (actual >= expected_floor).all(), (
        f"some leaf is below the min_samples_leaf={frac} floor "
        f"(expected ≥ {expected_floor}); got {actual.tolist()}"
    )


def test_min_samples_leaf_int_path_still_works(rng) -> None:
    """Integer min_samples_leaf preserved as the absolute floor."""
    n = 200
    X = rng.uniform(-1, 1, size=(n, 2))
    y = X.sum(axis=1) + 0.05 * rng.standard_normal(n)
    model = LinearTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=20
    ).fit(X, y)
    leaf_n = np.asarray(model.tree_.leaf_uq.n)
    is_leaf = np.asarray(model.tree_.is_leaf)
    assert (leaf_n[is_leaf] >= 20).all()
