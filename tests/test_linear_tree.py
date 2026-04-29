"""Smoke tests for LinearTreeRegressor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt import LinearTreeRegressor


def test_fit_predict_smoke_1d(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=4, max_bins=10, min_samples_leaf=10)
    model.fit(X, y)
    yh = model.predict(X)
    assert yh.shape == (X.shape[0],)
    # should fit reasonably well — MAE < std(y)
    assert float(jnp.mean(jnp.abs(yh - y))) < float(np.std(y))


def test_constant_target_yields_constant_prediction(rng) -> None:
    X = rng.uniform(-1, 1, size=(80, 2))
    y = np.full(80, 7.5)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    yh = model.predict(X)
    np.testing.assert_allclose(np.asarray(yh), 7.5, atol=1e-3)


def test_apply_returns_leaf_ids_for_each_row(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    leaves = model.apply(X)
    assert leaves.shape == (X.shape[0],)
    # All assigned leaves must in fact be leaves in the tree
    is_leaf = np.asarray(model.tree_.is_leaf)
    assert is_leaf[np.asarray(leaves)].all()


def test_fit_branin_reasonable_quality(branin_2d) -> None:
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=6, max_bins=8, min_samples_leaf=15).fit(X, y)
    yh = model.predict(X)
    # Depth-6 LMDT on Branin reliably reaches R² ≈ 0.95+; the previous
    # 0.7 floor was looser than any plausible regression target. We
    # tighten to a band: lower bound catches real regressions, upper
    # bound flags suspiciously-perfect fits (e.g., over-deep trees that
    # interpolate noise).
    ss_res = float(jnp.sum((y - yh) ** 2))
    ss_tot = float(jnp.sum((y - jnp.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert 0.90 <= r2 <= 0.999, f"R^2 outside expected band: {r2:.3f}"


@pytest.mark.parametrize("criterion", ["mae", "rmse", "msle", "max_abs"])
def test_all_documented_criteria_run(branin_2d, criterion) -> None:
    X, y = branin_2d
    if criterion == "msle":
        # msle requires positive y; shift Branin so it's strictly > 0.
        y = y - y.min() + 1.0
    model = LinearTreeRegressor(
        criterion=criterion, max_depth=3, max_bins=4, min_samples_leaf=20
    ).fit(X, y)
    yh = model.predict(X)
    assert yh.shape == (X.shape[0],)
    # Tree should fit better than a constant predictor.
    mae_const = float(jnp.mean(jnp.abs(y - jnp.mean(y))))
    mae_tree = float(jnp.mean(jnp.abs(yh - y)))
    assert mae_tree < mae_const, f"{criterion}: tree no better than constant"


def test_unknown_criterion_raises(toy_1d) -> None:
    X, y = toy_1d
    with pytest.raises(ValueError, match="Unknown criterion"):
        LinearTreeRegressor(criterion="not_a_real_loss").fit(X, y)


def test_predict_tree_apply_tree_reexports(toy_1d) -> None:
    """`predict_tree` and `apply_tree` should be importable from the
    top-level `jax_ldt` package and produce the same result as the
    underlying internal symbols."""
    import numpy as np
    from jax_ldt import predict_tree, apply_tree, LinearTreeRegressor
    from jax_ldt import tree_core

    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=10).fit(X, y)

    yh_top = np.asarray(predict_tree(model.tree_, X))
    yh_int = np.asarray(tree_core.predict(model.tree_, X))
    np.testing.assert_array_equal(yh_top, yh_int)

    leaves_top = np.asarray(apply_tree(model.tree_, X))
    leaves_int = np.asarray(tree_core.apply_(model.tree_, X))
    np.testing.assert_array_equal(leaves_top, leaves_int)


def test_multi_target_predict_shape(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(120, 2)).astype(np.float64)
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]]).astype(np.float64)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    yh = model.predict(X)
    assert yh.shape == (120, 2)
