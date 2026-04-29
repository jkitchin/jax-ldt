"""Optional sklearn-compatible wrapper around :class:`LinearTreeRegressor`.

This module is opt-in: importing it requires scikit-learn at runtime.
The core jax-ldt package has no sklearn dependency. The wrapper exists
for users who want to plug a jax-ldt regressor into sklearn machinery
such as :func:`sklearn.model_selection.cross_val_score`,
:class:`sklearn.pipeline.Pipeline`, or
:class:`sklearn.model_selection.GridSearchCV`.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from jax_ldt.linear_tree import LinearTreeRegressor


def _require_sklearn():
    try:
        import sklearn  # type: ignore  # noqa: F401
        from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by import-guard test
        raise ImportError(
            "jax_ldt.sklearn_compat requires scikit-learn. Install it with "
            "`pip install scikit-learn`."
        ) from exc
    return BaseEstimator, RegressorMixin


# Resolve the sklearn base classes at import time. We accept the cost of
# this eager import (this module is itself opt-in) so the resulting class
# inherits from the genuine sklearn bases — which is what
# `check_estimator`, `cross_val_score`, and friends look for.
BaseEstimator, RegressorMixin = _require_sklearn()


class SklearnLinearTreeRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around :class:`LinearTreeRegressor`.

    Useful for :func:`sklearn.model_selection.cross_val_score`,
    :class:`sklearn.pipeline.Pipeline`,
    :class:`sklearn.model_selection.GridSearchCV`, and
    :func:`sklearn.base.clone`.

    All hyperparameters of the underlying :class:`LinearTreeRegressor`
    are forwarded as constructor keyword arguments.

    Notes
    -----
    Some strict :func:`sklearn.utils.estimator_checks.check_estimator`
    contracts are not satisfied:

    - We do not implement the public ``feature_names_in_`` machinery.
    - The fitted attribute introspection (``__sklearn_is_fitted__``)
      is light-weight; tests that probe estimator state in detail may
      need to be skipped.

    These omissions are limited to non-numeric API surface; the basic
    ``fit/predict/score/get_params/set_params/clone`` contract is
    fully supported.
    """

    def __init__(
        self,
        criterion: str = "mae",
        max_depth: int = 32,
        min_samples_split: Union[int, float] = 6,
        min_samples_leaf: Union[int, float] = 0.01,
        max_bins: int = 25,
        min_impurity_decrease: float = 0.0,
        categorical_features: Optional[tuple] = None,
        split_features: Optional[tuple] = None,
        linear_features: Optional[tuple] = None,
        ridge: float = 1e-5,
        depth_first: bool = True,
    ) -> None:
        # sklearn requires that __init__ stores every constructor arg
        # under the same attribute name without modification, so that
        # `clone(self)` and `get_params(deep=True)` work correctly.
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.ridge = ridge
        self.depth_first = depth_first

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLinearTreeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._estimator = LinearTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            min_impurity_decrease=self.min_impurity_decrease,
            categorical_features=self.categorical_features,
            split_features=self.split_features,
            linear_features=self.linear_features,
            ridge=self.ridge,
            depth_first=self.depth_first,
        )
        self._estimator.fit(X, y)
        self.n_features_in_ = self._estimator.n_features_in_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_estimator"):
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.asarray(self._estimator.predict(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # R² (coefficient of determination), matching sklearn convention:
        #   ss_tot == 0 (constant target) and ss_res == 0  → 1.0 (perfect)
        #   ss_tot == 0 and ss_res != 0                    → 0.0
        # See sklearn.metrics.r2_score for the canonical definition.
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_pred = self.predict(X).reshape(-1)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Tag overrides for sklearn's estimator-check machinery.
    # sklearn 1.6 deprecated `_more_tags` in favour of
    # `__sklearn_tags__()`; both are implemented for cross-version
    # compatibility (sklearn falls back to whichever it knows about).
    # ------------------------------------------------------------------
    def _more_tags(self) -> dict[str, Any]:
        return {
            "requires_y": True,
            "allow_nan": False,
            "X_types": ["2darray"],
        }

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        # Construct the tags via the parent-provided API when available
        # (sklearn ≥ 1.6) and override only what we need.
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_estimator") and self._estimator is not None
