"""Boundary-input validation tests (G-7, G-16)."""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor


@pytest.mark.parametrize(
    "regressor",
    [LinearTreeRegressor, HyperplaneTreeRegressor],
)
class TestFitValidation:
    def test_empty_X_raises(self, regressor) -> None:
        X = np.zeros((0, 2), dtype=np.float64)
        y = np.zeros((0,), dtype=np.float64)
        with pytest.raises(ValueError, match="at least 1 sample"):
            regressor().fit(X, y)

    def test_mismatched_lengths_raise(self, regressor) -> None:
        X = np.zeros((10, 2), dtype=np.float64)
        y = np.zeros((9,), dtype=np.float64)
        with pytest.raises(ValueError, match="first-dim mismatch"):
            regressor().fit(X, y)

    def test_X_1d_raises(self, regressor) -> None:
        X = np.zeros(10, dtype=np.float64)
        y = np.zeros(10, dtype=np.float64)
        with pytest.raises(ValueError, match="2-D"):
            regressor().fit(X, y)

    def test_nan_in_X_raises(self, regressor, rng) -> None:
        X = rng.uniform(-1, 1, size=(20, 2))
        X[3, 0] = np.nan
        y = X.sum(axis=1)
        with pytest.raises(ValueError, match="non-finite"):
            regressor().fit(X, y)

    def test_inf_in_y_raises(self, regressor, rng) -> None:
        X = rng.uniform(-1, 1, size=(20, 2))
        y = X.sum(axis=1)
        y[0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            regressor().fit(X, y)


def test_predict_shape_mismatch_raises(rng) -> None:
    X = rng.uniform(-1, 1, size=(40, 3))
    y = X.sum(axis=1)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((5, 4)))  # wrong feature count


def test_float32_inputs_warn_then_run(rng) -> None:
    """G-15: float32 X/y trigger a UserWarning but the model still fits."""
    import warnings

    X = rng.uniform(-1.0, 1.0, size=(60, 2)).astype(np.float32)
    y = X.sum(axis=1).astype(np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LinearTreeRegressor(
            max_depth=3, max_bins=5, min_samples_leaf=10
        ).fit(X, y)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("Casting X" in m for m in messages), messages
    assert any("Casting y" in m for m in messages), messages

    yh = model.predict(X)
    assert yh.shape == (60,)
    assert np.all(np.isfinite(np.asarray(yh)))


def test_predict_nan_raises(rng) -> None:
    X = rng.uniform(-1, 1, size=(40, 2))
    y = X.sum(axis=1)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    X_bad = np.zeros((3, 2))
    X_bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(X_bad)
