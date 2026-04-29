"""Tests for linear-propagation and quadratic UQ."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt import LinearTreeRegressor
from jax_ldt.uncertainty import (
    LinearPropagationUQ,
    QuadraticUQ,
    calibration_metrics,
    linprop_uncertainty,
)


def test_linprop_returns_per_sample_uncertainty(branin_2d) -> None:
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=15).fit(X, y)
    sigma = linprop_uncertainty(model.tree_, X)
    assert sigma.shape == (X.shape[0],)
    assert jnp.all(sigma >= 0.0)


def test_linprop_uq_class_predict_matches_function(branin_2d) -> None:
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    s_func = linprop_uncertainty(model.tree_, X)
    s_cls = LinearPropagationUQ().predict(model.tree_, X)
    np.testing.assert_allclose(np.asarray(s_func), np.asarray(s_cls), atol=1e-12)


def test_linprop_uncertainty_grows_outside_training_distribution(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=4, max_bins=10, min_samples_leaf=15).fit(X, y)

    inside = np.array([[0.0]])  # near sample mean
    outside = np.array([[100.0]])  # far outside training range

    s_in = float(linprop_uncertainty(model.tree_, jnp.asarray(inside)).mean())
    s_out = float(linprop_uncertainty(model.tree_, jnp.asarray(outside)).mean())
    assert s_out > s_in, f"expected uncertainty to grow outside training: in={s_in}, out={s_out}"


def test_quadratic_uq_calibrates_and_predicts(branin_2d) -> None:
    """QuadraticUQ should produce non-degenerate σ on a non-linear target.

    Branin is strongly non-linear, so a linear-leaf tree leaves a real
    residual the quadratic correction can pick up — σ is **not** trivially
    zero everywhere. The previous version only asserted shape and ≥0,
    which would also pass for a no-op σ ≡ 0 implementation."""
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    quq = QuadraticUQ(ridge=1e-5).calibrate(model.tree_, X, y)
    sigma = quq.predict(model.tree_, X)
    assert sigma.shape == (X.shape[0],)
    assert jnp.all(sigma >= 0.0)
    # σ must be informative on at least some points — the residual from
    # the linear leaves on a non-linear target is non-zero. We compare
    # to the residual-MAE scale rather than an absolute threshold so the
    # test stays meaningful if we re-tune ridge / depth defaults.
    yh = np.asarray(model.predict(X))
    residual_scale = float(np.mean(np.abs(np.asarray(y) - yh)))
    sigma_max = float(jnp.max(sigma))
    assert sigma_max > 0.05 * residual_scale, (
        f"σ_max={sigma_max:.4g} is implausibly small relative to "
        f"residual scale {residual_scale:.4g} — quadratic correction may be a no-op"
    )


def test_linprop_constant_feature_does_not_blow_up() -> None:
    """G-11: a feature held constant inside a leaf must not inflate σ.

    We construct a 2D dataset where feature 1 is identically zero inside
    every leaf, then evaluate σ at a far-away test point. The old code
    floored ``x_var`` at ``1e-12`` which would give σ ~ 1e6 from a
    distance of 1.0. The fixed code drops the constant feature from the
    deviation sum entirely, so σ should be on the order of
    ``sqrt(mse) + sqrt(deviation_from_other_features)``.
    """
    rng = np.random.default_rng(7)
    n = 200
    X0 = rng.uniform(-1.0, 1.0, size=n)
    X1 = np.zeros(n)  # constant — leaf cannot learn from this
    X = np.column_stack([X0, X1]).astype(np.float64)
    y = (0.7 * X0 + 0.05 * rng.standard_normal(n)).astype(np.float64)

    model = LinearTreeRegressor(
        max_depth=2, max_bins=4, min_samples_leaf=20
    ).fit(X, y)

    # Test point with feature 1 far from its leaf-mean of 0.0.
    x_far = np.array([[0.0, 5.0]])
    sigma = float(linprop_uncertainty(model.tree_, jnp.asarray(x_far)).mean())

    assert np.isfinite(sigma), f"σ must be finite, got {sigma}"
    # Bound: sqrt(mse) over leaves is small (≤ a few × residual std).
    # Empirical residual std is ~0.05 here; allow generous slack but
    # certainly nowhere near the ~1e6 the old floor would produce.
    assert sigma < 5.0, f"σ at far-away constant feature blew up: {sigma}"


def test_linprop_single_sample_leaf_returns_sqrt_mse() -> None:
    """G-11: a 1-sample leaf has n=1; the deviation term must vanish.

    With ``min_samples_leaf=1`` and a tiny dataset the tree can put a
    single point in some leaves. Then ``n - 1 == 0`` and we cannot
    divide by it; the formula degenerates to ``sqrt(mse / n) = sqrt(mse)``
    (and ``mse`` itself is 0 for a 1-sample leaf, but we still expect a
    finite, sensible σ).
    """
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(8, 1)).astype(np.float64)
    y = np.sin(X[:, 0]).astype(np.float64)

    model = LinearTreeRegressor(
        max_depth=8, max_bins=10, min_samples_leaf=1, min_samples_split=2
    ).fit(X, y)

    # Walk every leaf and check at the leaf mean: σ must equal sqrt(mse / n).
    uq = model.tree_.leaf_uq
    is_leaf = np.asarray(model.tree_.is_leaf)
    for lid in np.where(is_leaf)[0]:
        n_leaf = float(uq.n[lid])
        if n_leaf < 1.0:
            continue
        mse = float(uq.mse[lid, 0])
        # Use a test point at the leaf mean so the deviation term would be 0
        # regardless. We are checking that 1-sample leaves give finite σ.
        x_mean_in = np.asarray(uq.x_mean[lid])[: model.n_features_in_]
        sigma = float(
            linprop_uncertainty(
                model.tree_, jnp.asarray(x_mean_in.reshape(1, -1))
            ).mean()
        )
        expected = np.sqrt(max(mse / max(n_leaf, 1.0), 0.0))
        assert np.isfinite(sigma)
        np.testing.assert_allclose(sigma, expected, atol=1e-8)


def test_quadratic_uq_zero_when_quadratic_recovers_target(rng) -> None:
    """If the leaf model is already the right linear model, |lin - quad| ≈ 0."""
    X = rng.uniform(-1, 1, size=(150, 2))
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 1.0  # exactly linear, should fit perfectly

    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=20).fit(X, y)
    quq = QuadraticUQ(ridge=1e-5).calibrate(model.tree_, X, y)
    sigma = quq.predict(model.tree_, X)
    assert float(jnp.mean(sigma)) < 0.05


def test_calibration_metrics_lazy_import(rng) -> None:
    pytest.importorskip("uncertainty_toolbox")
    n = 80
    y_true = rng.standard_normal(n).astype(np.float64)
    y_pred = (y_true + 0.1 * rng.standard_normal(n)).astype(np.float64)
    sigma = np.full(n, 0.15, dtype=np.float64)
    out = calibration_metrics(y_pred, sigma, y_true)
    for key in ("accuracy", "sharpness", "calibration", "scoring_rule"):
        assert key in out
