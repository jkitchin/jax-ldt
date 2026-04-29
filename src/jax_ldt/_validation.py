"""Boundary input validation shared by the regressors.

These checks live here (not in `tree_core`) so the JIT-compiled inner
kernels stay free of Python-level shape / dtype branching.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np


def _maybe_warn_non_float64(arr, name: str) -> None:
    """Emit a UserWarning when ``arr`` is not float64.

    jax-ldt always operates in 64-bit (see ``conftest.py`` enabling
    ``jax_enable_x64``). Float32 inputs would silently lose precision
    in the ridge solver and split-finder, so we warn the caller that
    the array is being upcast.
    """
    dtype = getattr(arr, "dtype", None)
    if dtype is None:
        return
    if dtype != jnp.float64:
        warnings.warn(
            f"Casting {name} from {dtype} to float64; jax-ldt operates in 64-bit.",
            UserWarning,
            stacklevel=3,
        )


def _all_finite(arr) -> bool:
    """Return True iff every element of ``arr`` is finite.

    Uses numpy on a host view to avoid a per-fit device→host scalar
    transfer (`bool(jnp.all(...))`) which is brutal under cross-validation
    or active-learning loops.
    """
    return bool(np.isfinite(np.asarray(arr)).all())


def _validate_fit_inputs(X: jnp.ndarray, y: jnp.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {tuple(X.shape)}")
    if X.shape[0] == 0:
        raise ValueError("X must have at least 1 sample.")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D; got shape {tuple(y.shape)}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y first-dim mismatch: X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
        )
    if not _all_finite(X):
        raise ValueError("X contains non-finite values (NaN or inf).")
    if not _all_finite(y):
        raise ValueError("y contains non-finite values (NaN or inf).")
    _maybe_warn_non_float64(X, "X")
    _maybe_warn_non_float64(y, "y")


def _validate_predict_input(X: jnp.ndarray, n_features_in: int) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {tuple(X.shape)}")
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features but model was fitted on {n_features_in}."
        )
    if not _all_finite(X):
        raise ValueError("X contains non-finite values (NaN or inf).")
