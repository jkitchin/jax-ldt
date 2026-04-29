"""Pin down the design rules independently of upstream behaviour.

These tests don't compare to the reference PyTorch implementation —
they verify that the choices we documented in the plan are reflected
in the code. They serve as regression guards if anyone refactors.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt.hyperplanes import generate_planes_to_index, symmetrize
from jax_ldt.linear_regression import add_bias, fit_ridge


def test_dedup_rows_are_lex_sorted_after_rounding() -> None:
    """generate_planes_to_index returns lex-sorted, dedup'd rows at given precision."""
    rows = generate_planes_to_index(2, max_weight=3, tol_decimals=4)
    arr = np.asarray(rows)
    # lex sort: each row's first differing element from the prev should be larger
    for i in range(1, len(arr)):
        assert tuple(arr[i]) > tuple(arr[i - 1])


def test_argmin_breaks_ties_to_lowest_flat_index() -> None:
    """jnp.argmin on tied values picks the lowest flat index."""
    M = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    flat_idx = int(jnp.argmin(M))
    # All four entries tie; argmin returns the first.
    assert flat_idx == 0


def test_ridge_added_before_solve_not_after() -> None:
    """For a rank-deficient X, ridge applied BEFORE the solve gives a finite β.

    If we solved (XᵀX) β = Xᵀy without ridge first and then added
    ridge to β somehow, the solve would fail or yield NaN. Adding ridge
    to XᵀX *before* the solve is the correct (and tested) order.
    """
    # Two perfectly collinear samples
    X = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = jnp.array([2.0, 4.0, 6.0])
    X_aug = add_bias(X)
    params = fit_ridge(X_aug, y, ridge=1e-3)
    assert jnp.all(jnp.isfinite(params))


def test_symmetrize_normalises_first_nonzero_to_one() -> None:
    LCs = jnp.array([[2.0, 4.0], [-1.0, 1.0]])
    out = symmetrize(LCs)
    arr = np.asarray(out)
    # Every row's first nonzero element should equal 1 (after normalisation).
    for row in arr:
        first_nz = row[row != 0][0]
        assert abs(first_nz - 1.0) < 1e-6, f"row {row} first nonzero {first_nz} != 1"


def test_quantile_bin_count_matches_max_bins_minus_one() -> None:
    """thresholds = quantile(linspace(0, 1, max_bins+1))[1:-1] gives max_bins-1 cuts."""
    from jax_ldt.tree_core import _compute_thresholds

    X = jnp.linspace(0, 10, 100).reshape(-1, 1)
    thr = _compute_thresholds(X, max_bins=10)
    assert thr.shape == (9, 1)


def test_leaf_fit_deterministic_on_same_data(rng) -> None:
    """Refitting the exact same leaf rows yields bitwise-identical params."""
    X = rng.uniform(-1, 1, size=(50, 3))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(50)
    X_aug = add_bias(jnp.asarray(X))

    p1 = fit_ridge(X_aug, jnp.asarray(y), ridge=1e-5)
    p2 = fit_ridge(X_aug, jnp.asarray(y), ridge=1e-5)
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))
