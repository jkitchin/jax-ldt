"""Tests for the JAX ridge linear-regression solver."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt.linear_regression import add_bias, fit_ridge, fit_ridge_weighted, predict


def test_fit_ridge_recovers_known_line(rng) -> None:
    X = rng.uniform(-1, 1, size=(50, 2))
    true_intercept = 0.7
    true_w = np.array([1.5, -0.5])
    y = X @ true_w + true_intercept

    X_aug = add_bias(jnp.asarray(X))
    params = fit_ridge(X_aug, jnp.asarray(y), ridge=1e-8)
    np.testing.assert_allclose(np.asarray(params).ravel(), [true_intercept, *true_w], atol=1e-4)


def test_fit_ridge_handles_rank_deficient() -> None:
    """X with a duplicated column would be singular without ridge."""
    X = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = jnp.array([2.0, 4.0, 6.0])
    X_aug = add_bias(X)
    # Should not blow up.
    params = fit_ridge(X_aug, y, ridge=1e-3)
    yh = predict(X_aug, params)
    np.testing.assert_allclose(np.asarray(yh), [2.0, 4.0, 6.0], atol=0.05)


def test_fit_ridge_weighted_zero_weight_row_is_ignored() -> None:
    X = jnp.array([[0.0], [1.0], [10.0]])
    y = jnp.array([0.0, 1.0, 999.0])
    w = jnp.array([1.0, 1.0, 0.0])
    X_aug = add_bias(X)
    params = fit_ridge_weighted(X_aug, y, w, ridge=1e-8)
    # the (10, 999) row should be ignored
    yh_first_two = predict(X_aug[:2], params)
    np.testing.assert_allclose(np.asarray(yh_first_two), [0.0, 1.0], atol=1e-3)


def test_predict_squeezes_single_target() -> None:
    X_aug = jnp.array([[1.0, 2.0], [1.0, 3.0]])
    params = jnp.array([[0.0], [1.0]])
    yh = predict(X_aug, params)
    assert yh.shape == (2,)
