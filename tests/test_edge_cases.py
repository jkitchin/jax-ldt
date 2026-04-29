"""Boundary / pathological-input coverage.

Each test pins a specific edge case the rest of the suite previously
under-exercised: degenerate input shapes, NaN, all-equal features,
non-contiguous arrays, all-zero weights, extreme magnitudes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor
from jax_ldt.linear_regression import add_bias, fit_ridge_weighted, predict


# ---------------------------------------------------------------------------
# Degenerate input shapes
# ---------------------------------------------------------------------------


def test_lmdt_rejects_zero_row_X() -> None:
    """Empty X must fail at the validation boundary, not deep inside fit."""
    X = np.zeros((0, 2), dtype=np.float64)
    y = np.zeros((0,), dtype=np.float64)
    with pytest.raises(ValueError, match="at least 1 sample"):
        LinearTreeRegressor(max_depth=2).fit(X, y)


def test_lmdt_single_row_fit_runs() -> None:
    """A 1-row dataset is degenerate but must not crash; the resulting
    tree predicts the single y close to exactly (ridge regularisation
    shrinks the leaf parameters very slightly toward zero)."""
    X = np.array([[0.5, -0.5]], dtype=np.float64)
    y = np.array([3.0], dtype=np.float64)
    model = LinearTreeRegressor(
        max_depth=2, max_bins=3, min_samples_leaf=1, min_samples_split=2
    ).fit(X, y)
    yh = np.asarray(model.predict(X))
    # Default ridge=1e-5 produces O(1e-5) shrinkage of the bias.
    np.testing.assert_allclose(yh, [3.0], atol=1e-3)


def test_xy_length_mismatch_raises() -> None:
    X = np.zeros((10, 2), dtype=np.float64)
    y = np.zeros((9,), dtype=np.float64)
    with pytest.raises(ValueError, match="first-dim mismatch"):
        LinearTreeRegressor(max_depth=2).fit(X, y)


# ---------------------------------------------------------------------------
# NaN / inf rejection
# ---------------------------------------------------------------------------


def test_nan_in_X_raises_at_fit(rng) -> None:
    X = rng.uniform(-1, 1, size=(20, 2)).astype(np.float64)
    X[3, 1] = np.nan
    y = X.sum(axis=1)
    with pytest.raises(ValueError, match="non-finite"):
        LinearTreeRegressor(max_depth=2).fit(X, y)


def test_nan_in_y_raises_at_fit(rng) -> None:
    X = rng.uniform(-1, 1, size=(20, 2)).astype(np.float64)
    y = X.sum(axis=1)
    y[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        LinearTreeRegressor(max_depth=2).fit(X, y)


def test_inf_in_X_raises_at_predict(rng) -> None:
    X = rng.uniform(-1, 1, size=(30, 2)).astype(np.float64)
    y = X.sum(axis=1)
    model = LinearTreeRegressor(max_depth=2, min_samples_leaf=5).fit(X, y)
    bad = np.array([[0.0, np.inf]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(bad)


# ---------------------------------------------------------------------------
# Pathological data distributions
# ---------------------------------------------------------------------------


def test_all_equal_X_does_not_split(rng) -> None:
    """When X is constant, no split can reduce loss; the tree must
    return a single root leaf rather than crash on degenerate quantiles."""
    X = np.full((40, 2), 0.5, dtype=np.float64)
    y = rng.standard_normal(40).astype(np.float64)
    model = LinearTreeRegressor(
        max_depth=4, max_bins=5, min_samples_leaf=5
    ).fit(X, y)
    is_leaf = np.asarray(model.tree_.is_leaf)
    # The whole tree should reduce to a single root leaf.
    assert is_leaf.sum() == 1
    assert is_leaf[0]


def test_y_with_ties_runs(rng) -> None:
    """Many duplicated y values must not break the criterion or the
    quantile threshold computation."""
    X = rng.uniform(-1, 1, size=(60, 1)).astype(np.float64)
    y = np.where(X[:, 0] > 0, 1.0, 0.0).astype(np.float64)  # only two values
    model = LinearTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=5
    ).fit(X, y)
    yh = np.asarray(model.predict(X))
    # MAE on a clean step at 0 should be small.
    assert float(np.mean(np.abs(yh - y))) < 0.1


def test_extreme_magnitudes_fit_predict(rng) -> None:
    """Features and y at 1e10 should fit and predict without overflow."""
    X = rng.uniform(-1.0, 1.0, size=(50, 2)).astype(np.float64) * 1e10
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float64)
    model = LinearTreeRegressor(
        max_depth=2, max_bins=4, min_samples_leaf=10
    ).fit(X, y)
    yh = np.asarray(model.predict(X))
    assert np.isfinite(yh).all()
    # Linear-leaf model should fit a linear target well even at scale.
    assert float(np.mean(np.abs(yh - y))) / float(np.std(y)) < 0.1


# ---------------------------------------------------------------------------
# Non-contiguous inputs
# ---------------------------------------------------------------------------


def test_non_contiguous_X_predicts_consistently(rng) -> None:
    """A strided view (`X[::2]`) must produce the same predictions as a
    contiguous copy — JAX/numpy interop should not silently mis-align."""
    X = rng.uniform(-1, 1, size=(80, 2)).astype(np.float64)
    y = X.sum(axis=1) + 0.05 * rng.standard_normal(80)
    model = LinearTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=10
    ).fit(X, y)

    view = X[::2]  # non-contiguous strided view
    contiguous = np.ascontiguousarray(view)
    yh_view = np.asarray(model.predict(view))
    yh_contig = np.asarray(model.predict(contiguous))
    np.testing.assert_array_equal(yh_view, yh_contig)


# ---------------------------------------------------------------------------
# Weighted ridge — all-zero / mixed-weight paths
# ---------------------------------------------------------------------------


def test_fit_ridge_weighted_all_zero_weights_does_not_nan() -> None:
    """All-zero weights are pathological but must not produce NaN;
    the ridge regulariser keeps the system positive-definite even when
    `XᵀWX = 0`. The fit collapses toward the regulariser's null
    solution (all-zero coefficients)."""
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = jnp.array([0.5, 1.0, 2.0])
    w = jnp.zeros(3)
    X_aug = add_bias(X)
    params = fit_ridge_weighted(X_aug, y, w, ridge=1e-3)
    assert jnp.all(jnp.isfinite(params))
    # With all weights zero, the only signal is the ridge prior at zero,
    # so coefficients should be small.
    assert float(jnp.max(jnp.abs(params))) < 1e-6


def test_fit_ridge_weighted_zero_weight_row_ignored() -> None:
    """Already covered indirectly by test_linear_regression.py; keep here
    as a regression guard for the public ``fit_ridge_weighted`` path."""
    X = jnp.array([[0.0], [1.0], [10.0]])
    y = jnp.array([0.0, 1.0, 999.0])
    w = jnp.array([1.0, 1.0, 0.0])
    X_aug = add_bias(X)
    params = fit_ridge_weighted(X_aug, y, w, ridge=1e-8)
    yh = predict(X_aug[:2], params)
    np.testing.assert_allclose(np.asarray(yh), [0.0, 1.0], atol=1e-3)


# ---------------------------------------------------------------------------
# Hyperplane tree on the same edge cases
# ---------------------------------------------------------------------------


def test_ht_handles_single_row() -> None:
    X = np.array([[0.5, -0.5]], dtype=np.float64)
    y = np.array([3.0], dtype=np.float64)
    model = HyperplaneTreeRegressor(
        max_depth=2,
        max_bins=3,
        min_samples_leaf=1,
        min_samples_split=2,
        max_weight=1,
        num_terms=2,
    ).fit(X, y)
    yh = np.asarray(model.predict(X))
    np.testing.assert_allclose(yh, [3.0], atol=1e-3)
