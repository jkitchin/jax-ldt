"""Tests for the optional sklearn-compatible wrapper.

Guarded by `pytest.importorskip("sklearn")` so the test suite still
runs in environments without scikit-learn installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_data(rng):
    X = rng.uniform(-2, 2, size=(120, 3))
    y = X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.05 * rng.standard_normal(120)
    return X.astype(np.float64), y.astype(np.float64)


def test_cross_val_score_returns_finite_scores(rng) -> None:
    pytest.importorskip("sklearn")
    from sklearn.model_selection import cross_val_score

    from jax_ldt import SklearnLinearTreeRegressor

    X, y = _make_data(rng)
    model = SklearnLinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=8)
    scores = cross_val_score(model, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_grid_search_cv(rng) -> None:
    pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV

    from jax_ldt import SklearnLinearTreeRegressor

    X, y = _make_data(rng)
    base = SklearnLinearTreeRegressor(max_bins=6, min_samples_leaf=8)
    grid = GridSearchCV(base, {"max_depth": [3, 5]}, cv=2)
    grid.fit(X, y)
    # The previous "in {3, 5}" assertion was vacuous (both options pass).
    # Validate that GridSearchCV actually drives the wrapper end-to-end:
    # cv_results_ has one row per candidate, the best estimator is
    # fitted, and its predictions are sensibly close to y on the train
    # set (the underlying tree is non-trivial).
    assert hasattr(grid, "best_params_")
    assert grid.best_params_["max_depth"] in {3, 5}
    assert len(grid.cv_results_["params"]) == 2
    # best_estimator_ must be a fitted clone, not the same instance
    assert grid.best_estimator_ is not base
    assert grid.best_estimator_.n_features_in_ == X.shape[1]
    yh = grid.best_estimator_.predict(X)
    # MAE on a y with std ~1 should be well under 1.0; the alternative
    # of "did anything fit at all" was previously checked only via
    # `hasattr(grid, "best_params_")`.
    assert float(np.mean(np.abs(y - yh))) < float(np.std(y))


def test_clone_round_trip(rng) -> None:
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    from jax_ldt import SklearnLinearTreeRegressor

    X, y = _make_data(rng)
    model = SklearnLinearTreeRegressor(
        max_depth=4, max_bins=6, min_samples_leaf=8
    ).fit(X, y)
    yh_orig = model.predict(X)
    yh_clone = clone(model).fit(X, y).predict(X)
    np.testing.assert_allclose(np.asarray(yh_clone), np.asarray(yh_orig), atol=1e-10)
