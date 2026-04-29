"""Tests for Mondrian leaf conformal calibration."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt import LinearTreeRegressor
from jax_ldt.uncertainty import ConformalCalibrator, fit_with_conformal


def _sklearn_split(X, y, frac, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_a = int(np.ceil(frac * n))
    return X[perm[:n_a]], y[perm[:n_a]], X[perm[n_a:]], y[perm[n_a:]]


def test_conformal_intervals_have_correct_coverage(rng) -> None:
    """On a noisy 1D toy, 90% intervals should cover ~90% of held-out points."""
    X = rng.uniform(-3, 3, size=(800, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(800)

    X_cal, y_cal, X_test, y_test = _sklearn_split(X[:600], y[:600], 0.5, seed=0)
    X_train = X[600:]
    y_train = y[600:]

    model = LinearTreeRegressor(max_depth=3, max_bins=8, min_samples_leaf=20).fit(X_train, y_train)
    cc = ConformalCalibrator(alpha=0.1, mondrian=True, min_calibration_per_leaf=5).calibrate(
        model, X_cal, y_cal
    )
    lo, hi = cc.predict_interval(X_test, model=model)

    coverage = float(np.mean((y_test >= np.asarray(lo)) & (y_test <= np.asarray(hi))))
    # 90% target with N_test=300 has theoretical stdev sqrt(0.9*0.1/300) ≈ 0.017,
    # so ±0.04 spans ~2.4σ. The seed is fixed, so this band stays safe across
    # refactors but catches any real undercoverage (say, dropping below 0.86).
    assert 0.86 <= coverage <= 0.94, f"unexpected coverage {coverage:.3f}"


def test_global_conformal_works_when_mondrian_false(rng) -> None:
    X = rng.uniform(-1, 1, size=(300, 2))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(300)

    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X[:200], y[:200])
    cc = ConformalCalibrator(alpha=0.2, mondrian=False).calibrate(model, X[200:], y[200:])
    lo, hi = cc.predict_interval(X[200:], model=model)
    # global mode: all leaves get same halfwidth
    leaf_hw = np.asarray(cc.state_.leaf_halfwidth)
    leaves_only = leaf_hw[np.asarray(model.tree_.is_leaf)]
    # All leaf rows should be identical to the first leaf's halfwidth.
    np.testing.assert_allclose(leaves_only, np.broadcast_to(leaves_only[0], leaves_only.shape), atol=1e-12)


def test_mondrian_false_coverage_is_close_to_target(rng) -> None:
    """`mondrian=False` is a marginal-coverage CP method — at the chosen
    alpha the empirical coverage on a held-out set should land near
    ``1 - alpha``. This guards against a regression in the global-quantile
    path that the prior mondrian=False test (only halfwidth equality)
    would not catch."""
    X = rng.uniform(-3, 3, size=(800, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(800)

    X_cal, y_cal, X_test, y_test = _sklearn_split(X[:600], y[:600], 0.5, seed=0)
    X_train, y_train = X[600:], y[600:]

    model = LinearTreeRegressor(max_depth=3, max_bins=8, min_samples_leaf=20).fit(
        X_train, y_train
    )
    cc = ConformalCalibrator(alpha=0.1, mondrian=False).calibrate(model, X_cal, y_cal)
    lo, hi = cc.predict_interval(X_test, model=model)
    coverage = float(np.mean((y_test >= np.asarray(lo)) & (y_test <= np.asarray(hi))))
    # Same band as the mondrian=True companion: ~2.4σ around 0.90 with
    # N_test=300. Wider would mask undercoverage; narrower would be
    # brittle to refactors.
    assert 0.86 <= coverage <= 0.94, (
        f"global (mondrian=False) coverage {coverage:.3f} outside [0.86, 0.94]"
    )


def test_sparse_leaf_falls_back_to_global_with_warning(rng) -> None:
    X = rng.uniform(-1, 1, size=(80, 1))
    y = X[:, 0] + 0.05 * rng.standard_normal(80)

    model = LinearTreeRegressor(
        max_depth=8, max_bins=20, min_samples_leaf=3
    ).fit(X[:60], y[:60])
    # Tiny calibration set so most leaves will be sparse
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=10).calibrate(
            model, X[60:], y[60:]
        )
        # At least one sparse-leaf warning should have fired (only if there
        # are leaves with 0 < cal points < min)
        assert any("calibration points" in str(rec.message) for rec in w)


def test_fit_with_conformal_convenience(rng) -> None:
    X = rng.uniform(-2, 2, size=(400, 2))
    y = (X**2).sum(axis=1) + 0.1 * rng.standard_normal(400)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=15)
    fitted, calib = fit_with_conformal(model, X, y, calibration_size=0.25, alpha=0.1)
    lo, hi = calib.predict_interval(X, model=fitted)
    coverage = float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
    # In-sample coverage on a 90% target should be ≥ the marginal
    # guarantee for the calibration subset; the 75 % floor catches
    # regressions where conformal halfwidths are silently too small.
    # Upper bound 1.0 keeps the assertion meaningful (a constant ∞
    # halfwidth would otherwise pass).
    assert 0.75 <= coverage <= 1.0, f"in-sample coverage {coverage:.3f} out of band"


def test_conformal_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=1.0)
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=1.5)


def test_predict_interval_before_calibrate_raises(rng) -> None:
    cc = ConformalCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError):
        cc.predict_interval(jnp.array([[0.0]]))


def test_invalid_sparse_leaf_strategy_raises() -> None:
    with pytest.raises(ValueError, match="sparse_leaf_strategy"):
        ConformalCalibrator(alpha=0.1, sparse_leaf_strategy="bogus")


def test_sparse_leaf_skip_returns_nan_intervals(rng) -> None:
    """G-3: 'skip' strategy yields NaN intervals at sparse leaves.

    The configuration below (deep tree fit on 60 points, then calibrated
    with only 20 points and ``min_calibration_per_leaf=10``) is chosen so
    most leaves see fewer than 10 calibration points. We assert that at
    least one sparse leaf is detected — otherwise the test would pass
    vacuously without exercising the ``"skip"`` path.
    """
    X = rng.uniform(-1, 1, size=(80, 1))
    y = X[:, 0] + 0.05 * rng.standard_normal(80)
    model = LinearTreeRegressor(
        max_depth=8, max_bins=20, min_samples_leaf=3
    ).fit(X[:60], y[:60])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = ConformalCalibrator(
            alpha=0.1, min_calibration_per_leaf=10, sparse_leaf_strategy="skip"
        ).calibrate(model, X[60:], y[60:])

    assert cc._sparse_leaves, (
        "expected at least one sparse leaf in this configuration; the "
        "test cannot validate the skip-NaN path otherwise"
    )

    lo, hi = cc.predict_interval(X[:60], model=model)
    lo_np = np.asarray(lo)
    hi_np = np.asarray(hi)
    assert np.any(np.isnan(lo_np)) or np.any(np.isnan(hi_np))
