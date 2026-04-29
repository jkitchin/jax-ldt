"""Ridge-regularised least-squares for leaf and split regressions.

Two entry points:

- `fit_ridge(X, y, ridge)` — fit one regression. Returns params of shape
  `(n_features, n_targets)`. X is *augmented* (caller prepends bias).
- `fit_ridge_batched(X_batch, y_batch, ridge)` — vmapped over a leading
  batch axis; used to evaluate `K * B` candidate splits in parallel.

We solve the normal equations `(XᵀX + ridge·I) β = Xᵀy` because:
- The ridge regularisation already stabilises XᵀX, so a positive-definite
  solver (`jax.scipy.linalg.solve(..., assume_a='pos')`) is faster than
  `lstsq` and numerically stable when the original X is rank-deficient.
- The matrix shape is (n_aug, n_aug); regardless of how many samples,
  the solve is O(n_aug^3), so on big subdomains this is much cheaper
  than a per-sample lstsq.

Ridge is added BEFORE the solve (ridge·I is folded into XᵀX). This is
the order assumed by upstream and pinned by `test_principled_choices.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


def add_bias(X: jnp.ndarray) -> jnp.ndarray:
    """Prepend a column of ones to X.

    X may be (..., n_samples, n_features); the bias column is added on
    the last axis.
    """
    ones = jnp.ones(X.shape[:-1] + (1,), dtype=X.dtype)
    return jnp.concatenate([ones, X], axis=-1)


def fit_ridge(
    X_aug: jnp.ndarray,
    y: jnp.ndarray,
    ridge: float,
) -> jnp.ndarray:
    """Fit ridge least-squares on already-augmented X.

    X_aug : (n_samples, n_aug)  -- caller has prepended the bias column
    y     : (n_samples,) or (n_samples, n_targets)
    ridge : nonnegative scalar

    Returns
    -------
    params : (n_aug, n_targets)
    """
    if y.ndim == 1:
        y = y[:, None]
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    eye = jnp.eye(XtX.shape[0], dtype=XtX.dtype)
    XtX = XtX + ridge * eye
    return jsl.solve(XtX, Xty, assume_a="pos")


def fit_ridge_weighted(
    X_aug: jnp.ndarray,
    y: jnp.ndarray,
    sample_weight: jnp.ndarray,
    ridge: float,
) -> jnp.ndarray:
    """Fit weighted ridge least-squares.

    sample_weight : (n_samples,) — nonnegative weights.
    """
    if y.ndim == 1:
        y = y[:, None]
    w = sample_weight[:, None]
    Xw = X_aug * w
    XtX = Xw.T @ X_aug
    Xty = Xw.T @ y
    eye = jnp.eye(XtX.shape[0], dtype=XtX.dtype)
    XtX = XtX + ridge * eye
    return jsl.solve(XtX, Xty, assume_a="pos")


# Vmapped over the leading batch axis. We use this to evaluate (K, B)
# candidate splits in one fused kernel: for each candidate, the masked
# data forms one element of the batch.
fit_ridge_batched = jax.vmap(fit_ridge, in_axes=(0, 0, None))


def predict(X_aug: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """X_aug @ params; if params is (n_aug, 1), squeeze trailing axis."""
    out = X_aug @ params
    if out.ndim > 1 and out.shape[-1] == 1:
        out = out[..., 0]
    return out
