"""Axis-aligned Linear Model Decision Tree (LMDT).

Thin wrapper around `tree_core.grow_tree` with an sklearn-flavoured but
sklearn-free fit/predict API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import jax.numpy as jnp
import numpy as np

from jax_ldt._types import Tree
from jax_ldt import tree_core
from jax_ldt._validation import _validate_fit_inputs

# np.ndarray is generic in NumPy 2.x; in user code we accept anything
# array-like and re-materialise via `jnp.asarray`. `Any` keeps the
# regressor signatures permissive without ratcheting `--strict` checks
# all the way down to the user's input dtype.
ArrayLike = Union[np.ndarray[Any, Any], jnp.ndarray]


@dataclass
class LinearTreeRegressor:
    """Axis-aligned Linear Model Decision Tree (LMDT).

    Piecewise-linear regression surrogate: a binary decision tree where
    every leaf holds an ordinary ridge-regularised linear model and
    splits are axis-aligned (a single feature compared to a threshold).
    Pure-JAX port of LLNL's ``LinearTreeRegressor`` from
    ``hyperplanetree``; the JIT-compiled inner kernel evaluates all
    candidate splits in a single fused vmap. Inputs may be ``np.ndarray``
    or ``jnp.ndarray``; everything else is plain Python.

    Parameters
    ----------
    criterion : str, default ``"mae"``
        Split-loss criterion. One of ``"mae"``, ``"rmse"``, ``"msle"``,
        ``"max_abs"``.
    max_depth : int, default ``32``
        Maximum tree depth (root is depth 0). Practical sweet spot is
        5-8; deeper trees overfit before they help on real data.
    min_samples_split : int or float, default ``6``
        Minimum number of samples in a node required to attempt a
        split. Floats in ``(0, 1)`` are interpreted as fractions of the
        training-set size.
    min_samples_leaf : int or float, default ``0.01``
        Minimum number of samples per child leaf. Floats in ``(0, 1)``
        are fractions of the training-set size; a hard floor of 3 is
        applied to keep the leaf ridge fit well-posed.
    max_bins : int, default ``25``
        Quantile bins per split feature. ``max_bins - 1`` interior
        quantile cuts are evaluated as candidate thresholds.
    min_impurity_decrease : float, default ``0.0``
        Minimum loss reduction required to accept a split. ``0.0`` means
        any non-negative improvement is accepted; positive values prune.
    categorical_features : tuple of int, optional
        Column indices treated as categorical (excluded from splitting
        and from the linear leaf model).
    split_features : tuple of int, optional
        Column indices eligible for splitting. Defaults to every
        non-categorical feature.
    linear_features : tuple of int, optional
        Column indices included in the leaf linear model. Defaults to
        every non-categorical feature.
    ridge : float, default ``1e-5``
        Ridge regularisation added to ``XᵀWX`` before the per-leaf
        solve. Guarantees positive-definiteness even on near-empty
        masks during the vmapped split evaluation.
    depth_first : bool, default ``True``
        If ``True``, grow nodes depth-first; otherwise breadth-first.

    Attributes
    ----------
    tree_ : Tree
        Fitted tree pytree (parallel-array layout). ``None`` until
        :meth:`fit` is called.
    n_features_in_ : int
        Number of input features observed at fit time.
    n_targets_ : int
        Number of regression targets (``1`` for 1D ``y``, otherwise
        ``y.shape[1]``).
    """

    criterion: str = "mae"
    max_depth: int = 32
    min_samples_split: Union[int, float] = 6
    min_samples_leaf: Union[int, float] = 0.01
    max_bins: int = 25
    min_impurity_decrease: float = 0.0
    categorical_features: Optional[tuple[int, ...]] = None
    split_features: Optional[tuple[int, ...]] = None
    linear_features: Optional[tuple[int, ...]] = None
    ridge: float = 1e-5
    depth_first: bool = True

    # populated after fit
    tree_: Optional[Tree] = field(default=None, init=False, repr=False)
    n_features_in_: Optional[int] = field(default=None, init=False, repr=False)
    n_targets_: Optional[int] = field(default=None, init=False, repr=False)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearTreeRegressor":
        # Materialise as JAX arrays first (preserving dtype) so the
        # validator can warn on non-float64 inputs, then upcast.
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        _validate_fit_inputs(X, y)
        X = X.astype(jnp.float64)
        y = y.astype(jnp.float64)
        self.n_features_in_ = int(X.shape[1])
        self.n_targets_ = 1 if y.ndim == 1 else int(y.shape[1])
        self.tree_ = tree_core.grow_tree(
            X,
            y,
            transform_matrix=None,
            linear_features=self.linear_features,
            split_features=self.split_features,
            categorical_features=self.categorical_features,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            ridge=self.ridge,
            depth_first=self.depth_first,
        )
        return self

    def predict(self, X: ArrayLike) -> jnp.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = jnp.asarray(X, dtype=jnp.float64)
        from jax_ldt._validation import _validate_predict_input
        _validate_predict_input(X, self.n_features_in_)
        return tree_core.predict(self.tree_, X)

    def apply(self, X: ArrayLike) -> jnp.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Call fit() before apply().")
        X = jnp.asarray(X, dtype=jnp.float64)
        from jax_ldt._validation import _validate_predict_input
        _validate_predict_input(X, self.n_features_in_)
        return tree_core.apply_tree(self.tree_, X)

    @property
    def num_leaves(self) -> int:
        if self.tree_ is None:
            raise RuntimeError("Call fit() before querying num_leaves.")
        return self.tree_.n_leaves

    @property
    def n_leaves(self) -> int:
        """Alias for :attr:`num_leaves` (matches ``Tree.n_leaves``)."""
        return self.num_leaves
