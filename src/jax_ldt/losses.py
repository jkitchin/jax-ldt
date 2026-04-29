"""Loss functions used as split criteria.

All take `(y_true, y_pred)` of shape `(..., n_targets)` and return a
scalar (or per-batch loss along the leading axis if `dim` is set).
JIT-friendly; no Python control flow.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

LossFn = Callable[..., jnp.ndarray]


def _broadcast_weights(weights: jnp.ndarray, like: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Reshape `weights` (1-D, length = like.shape[dim]) to broadcast against `like`."""
    pos_dim = dim if dim >= 0 else like.ndim + dim
    shape = [1] * like.ndim
    shape[pos_dim] = -1
    return weights.reshape(shape)


def _safe_weighted_mean(num: jnp.ndarray, w: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Σ(num) / Σ(w), with Σ(w)==0 → 0 instead of NaN.

    Mirrors the zero-weight guards used by the inner-kernel weighted
    losses in :mod:`tree_core` so public and private paths agree.
    """
    sw = jnp.sum(w, axis=dim)
    safe_sw = jnp.where(sw > 0, sw, 1)
    return jnp.where(sw > 0, jnp.sum(num, axis=dim) / safe_sw, 0.0)


def mae(y: jnp.ndarray, yh: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
        dim: int = -1) -> jnp.ndarray:
    err = jnp.abs(y - yh)
    if weights is not None:
        w = _broadcast_weights(weights, err, dim)
        return _safe_weighted_mean(err * w, w, dim)
    return jnp.mean(err, axis=dim)


def rmse(y: jnp.ndarray, yh: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
         dim: int = -1) -> jnp.ndarray:
    sq = (y - yh) ** 2
    if weights is not None:
        w = _broadcast_weights(weights, sq, dim)
        return jnp.sqrt(_safe_weighted_mean(sq * w, w, dim))
    return jnp.sqrt(jnp.mean(sq, axis=dim))


def msle(y: jnp.ndarray, yh: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
         dim: int = -1) -> jnp.ndarray:
    eps = 1e-6
    sq = jnp.square(jnp.log10(jnp.clip(y, min=eps)) - jnp.log10(jnp.clip(yh, min=eps)))
    if weights is not None:
        w = _broadcast_weights(weights, sq, dim)
        return _safe_weighted_mean(sq * w, w, dim)
    return jnp.mean(sq, axis=dim)


def max_abs(y: jnp.ndarray, yh: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
            dim: int = -1) -> jnp.ndarray:
    err = jnp.abs(y - yh)
    if weights is not None:
        w = _broadcast_weights(weights, err, dim)
        err = err * w
    return jnp.max(err, axis=dim)


CRITERIA: dict[str, LossFn] = {
    "mae": mae,
    "rmse": rmse,
    "msle": msle,
    "max_abs": max_abs,
}


def resolve_criterion(criterion: str | LossFn) -> LossFn:
    if callable(criterion):
        return criterion
    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Pick one of {sorted(CRITERIA)} or pass a callable."
        )
    return CRITERIA[criterion]
