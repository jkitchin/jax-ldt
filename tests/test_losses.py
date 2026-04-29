"""Unit tests for loss functions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt.losses import CRITERIA, mae, max_abs, msle, resolve_criterion, rmse


def test_mae_basic() -> None:
    y = jnp.array([[1.0], [2.0], [3.0]])
    yh = jnp.array([[1.0], [2.5], [4.0]])
    np.testing.assert_allclose(mae(y, yh, dim=0), [0.5], rtol=1e-6)


def test_rmse_basic() -> None:
    y = jnp.array([[1.0], [2.0], [3.0]])
    yh = jnp.array([[2.0], [3.0], [4.0]])
    np.testing.assert_allclose(rmse(y, yh, dim=0), [1.0], rtol=1e-6)


def test_max_abs_basic() -> None:
    y = jnp.array([[1.0], [2.0], [3.0]])
    yh = jnp.array([[1.0], [3.0], [3.0]])
    np.testing.assert_allclose(max_abs(y, yh, dim=0), [1.0], rtol=1e-6)


def test_msle_clips_zero() -> None:
    """MSLE clips both inputs at 1e-6 before log10. With y == yh == 0
    both sides clip to the same value, so the per-target MSLE is exactly
    zero rather than NaN/inf."""
    y = jnp.array([[0.0], [1.0]])
    yh = jnp.array([[0.0], [1.0]])
    out = msle(y, yh, dim=0)
    assert jnp.all(jnp.isfinite(out))
    np.testing.assert_allclose(np.asarray(out), [0.0], atol=1e-12)


def test_weighted_mae() -> None:
    y = jnp.array([[1.0], [2.0], [3.0]])
    yh = jnp.array([[1.0], [3.0], [4.0]])
    w = jnp.array([1.0, 0.0, 1.0])
    # zero-weight middle row should not contribute
    np.testing.assert_allclose(mae(y, yh, weights=w, dim=0), [0.5], rtol=1e-6)


def test_resolve_criterion_accepts_string_and_callable() -> None:
    assert resolve_criterion("mae") is CRITERIA["mae"]
    f = lambda y, yh, **_: jnp.array(0.0)
    assert resolve_criterion(f) is f
