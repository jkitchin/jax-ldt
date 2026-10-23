"""Core split-and-grow algorithm for linear / hyperplane decision trees.

Design notes:

1. Tree growth is a Python while-loop over a queue of node ids; topology
   is data-dependent and not JIT-compatible. Splits and predictions ARE
   JIT-compiled.

2. To keep tensor shapes stable across all nodes (so JIT compiles once),
   we operate on the full (X_aug, y) and use a per-node `mask` indicating
   which rows are currently in the subdomain. Quantile thresholds for
   each subdomain are computed eagerly outside the JIT (since the active
   row count is variable).

3. For each candidate split (col, bin), we evaluate two ridge regressions
   (left and right side) using the mask × side-mask as a 0/1
   sample-weight. This is mathematically equivalent to row-selection
   followed by ordinary ridge, and vmap covers all K*B candidates at
   once.

4. The fit uses normal-equations + `jax.scipy.linalg.solve`. Ridge is
   added to XᵀWX BEFORE the solve. Ridge guarantees invertibility even
   when w is sparse / nearly all-zero, so invalid candidates produce some
   theta which we then discard via a validity mask. See
   `test_principled_choices.py`.

5. **Reduced-space candidate evaluation.** The fitting design matrix is
   always ``X_fit = Z @ B`` where ``Z = [1, X]`` has ``m = n_features_in
   + 1`` columns and ``B`` is the (bias-augmented, ``linear_features``-
   masked) transform matrix. Every lifted column is therefore a linear
   function of the *original* features, so the ``n_aug``-dimensional
   ridge fit has at most ``m`` degrees of freedom. Using the push-through
   identity

       (Bᵀ G B + λI)⁻¹ Bᵀ v  =  Bᵀ (G BBᵀ + λI)⁻¹ v

   with ``G = ZᵀWZ`` and ``v = ZᵀWy`` (both ``m``-sized), the per-
   candidate solve drops from ``O(n_aug³)`` to ``O(N·m² + m³)`` and the
   fitted values follow as ``ŷ = (Z·BBᵀ)·u``. This is an algebraic
   identity, not an approximation: the candidate losses agree with the
   dense fit to round-off (~1e-15 relative). It is what keeps hyperplane
   fit cost *linear* in the lifted-feature count instead of quartic —
   see `docs/benchmarks.md` ("Cost model") and issue #2.

6. **Bounded working set.** Candidates are evaluated in column chunks
   sized so the kernel's peak intermediate allocation stays under
   ``SPLIT_MEMORY_BUDGET_BYTES``. The chunk width is a compile-time
   constant for a given ``grow_tree`` call, so the kernel still traces
   once; the last chunk is padded (with ``+inf`` thresholds, which score
   as invalid) rather than reshaped, to avoid a second compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from jax_ldt._types import LeafUQ, Tree
from jax_ldt.linear_regression import add_bias


# --------------------------------------------------------------------------
# Internal Python-side node container, only used during tree growth.
# This is NOT a pytree; it gets converted to parallel arrays at the end.
# --------------------------------------------------------------------------
@dataclass
class _GrowNode:
    node_id: int
    is_leaf: bool
    feature: int  # -1 if leaf
    threshold: float  # NaN if leaf
    left: int  # -1 if leaf
    right: int  # -1 if leaf
    leaf_params: jax.Array  # (n_aug, n_targets); zeros for non-leaf
    # for UQ:
    leaf_x_mean: jax.Array  # (n_features,)
    leaf_x_var: jax.Array  # (n_features,)
    leaf_n: object  # device scalar for leaves, int 0 for placeholders
    leaf_mse: jax.Array  # (n_targets,)


# --------------------------------------------------------------------------
# Inner kernel: given a subdomain (mask) and candidate thresholds,
# evaluate every (split_col, threshold_bin) pair and return losses.
# --------------------------------------------------------------------------


def _fit_one_side(
    X_aug: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    ridge: float,
) -> jnp.ndarray:
    """Fit one ridge regression with sample weights.

    X_aug : (N, n_aug)
    y     : (N, T)
    w     : (N,) nonnegative sample weights (typically 0/1)
    Returns params (n_aug, T).
    """
    Xw = X_aug * w[:, None]
    XtX = Xw.T @ X_aug
    Xty = Xw.T @ y
    eye = jnp.eye(XtX.shape[0], dtype=XtX.dtype)
    return jsl.solve(XtX + ridge * eye, Xty, assume_a="pos")


_fit_many_sides = jax.vmap(_fit_one_side, in_axes=(None, None, 0, None))


def _weighted_mae(y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted MAE per target. y, yh: (N, T); w: (N,). Returns (T,)."""
    err = jnp.abs(y - yh) * w[:, None]
    sw = jnp.sum(w)
    safe_sw = jnp.where(sw > 0.0, sw, 1.0)
    out = jnp.sum(err, axis=0) / safe_sw
    return jnp.where(sw > 0.0, out, jnp.inf)


def _weighted_mse(y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted mean squared error per target. y, yh: (N, T); w: (N,). Returns (T,)."""
    err = ((y - yh) ** 2) * w[:, None]
    sw = jnp.sum(w)
    safe_sw = jnp.where(sw > 0.0, sw, 1.0)
    out = jnp.sum(err, axis=0) / safe_sw
    return jnp.where(sw > 0.0, out, jnp.inf)


def _weighted_rmse(y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    err = ((y - yh) ** 2) * w[:, None]
    sw = jnp.sum(w)
    safe_sw = jnp.where(sw > 0.0, sw, 1.0)
    out = jnp.sqrt(jnp.sum(err, axis=0) / safe_sw)
    return jnp.where(sw > 0.0, out, jnp.inf)


def _weighted_msle(y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted mean-squared log error. Clips inputs at 1e-6 before log10
    (matches upstream `criteria['msle']` convention)."""
    eps = 1e-6
    sq = (jnp.log10(jnp.clip(y, min=eps)) - jnp.log10(jnp.clip(yh, min=eps))) ** 2
    sq = sq * w[:, None]
    sw = jnp.sum(w)
    safe_sw = jnp.where(sw > 0.0, sw, 1.0)
    out = jnp.sum(sq, axis=0) / safe_sw
    return jnp.where(sw > 0.0, out, jnp.inf)


def _weighted_max_abs(y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Weighted max absolute error per target. Out-of-mask rows are masked to 0
    (effectively excluded from the max). Returns inf if no rows are active."""
    err = jnp.abs(y - yh) * w[:, None]
    sw = jnp.sum(w)
    out = jnp.max(err, axis=0)
    return jnp.where(sw > 0.0, out, jnp.inf)


_WEIGHTED_LOSS_FNS = {
    "mae": _weighted_mae,
    "mse": _weighted_mse,
    "rmse": _weighted_rmse,
    "msle": _weighted_msle,
    "max_abs": _weighted_max_abs,
}


def _weighted_loss(name: str):
    if name not in _WEIGHTED_LOSS_FNS:
        raise ValueError(
            f"Unknown criterion {name!r}. Pick one of {sorted(_WEIGHTED_LOSS_FNS)}."
        )
    return _WEIGHTED_LOSS_FNS[name]


def _reduced_yhat_one(
    Z: jnp.ndarray,  # (N, m)   = [1, X_raw]
    ZK: jnp.ndarray,  # (N, m)   = Z @ (B Bᵀ)
    BBt: jnp.ndarray,  # (m, m)  = B Bᵀ
    y: jnp.ndarray,  # (N, T)
    w: jnp.ndarray,  # (N,)
    ridge: float,
) -> jnp.ndarray:
    """Fitted values of one weighted ridge fit, computed in reduced space.

    Equivalent to ``X_fit @ _fit_one_side(X_fit, y, w, ridge)`` for
    ``X_fit = Z @ B``, but costs ``O(N·m² + m³)`` instead of
    ``O(N·n_aug² + n_aug³)``. See design note 5 in the module docstring
    for the identity. Returns ``(N, T)``.
    """
    Zw = Z * w[:, None]
    G = Zw.T @ Z  # (m, m)
    v = Zw.T @ y  # (m, T)
    eye = jnp.eye(BBt.shape[0], dtype=BBt.dtype)
    u = jsl.solve(G @ BBt + ridge * eye, v)  # (m, T)
    return ZK @ u


_reduced_yhat_many = jax.vmap(
    _reduced_yhat_one, in_axes=(None, None, None, None, 0, None)
)


def reduced_split_basis(
    X: jnp.ndarray,
    transform_matrix: jnp.ndarray,
    lin_cols: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Factor the fitting design matrix as ``X_fit = Z @ B``.

    ``X_fit`` is ``add_bias(X @ transform_matrix)`` with the columns
    outside ``lin_cols`` zeroed. Because ``add_bias(X @ A) == [1, X] @
    blockdiag(1, A)`` and the mask zeroes whole columns of that product,
    the factorisation is exact with ``Z = [1, X]`` of width
    ``m = n_features_in + 1``.

    Parameters
    ----------
    X : (N, n_features_in)
        Raw (untransformed) inputs.
    transform_matrix : (n_features_in, n_features_transformed)
        The hyperplane lift; pass the identity for an axis-aligned tree.
    lin_cols : (n_aug,) bool
        Which augmented columns take part in the fit (index 0 is bias).

    Returns
    -------
    Z : (N, m)      — ``[1, X]``
    BBt : (m, m)    — ``B Bᵀ``
    ZK : (N, m)     — ``Z @ B Bᵀ``, precomputed for the fitted values
    """
    X = jnp.asarray(X)
    n_in = int(X.shape[1])
    A = np.asarray(transform_matrix, dtype=np.float64)
    n_aug = A.shape[1] + 1
    lin_cols = np.asarray(lin_cols, dtype=bool)
    if lin_cols.shape != (n_aug,):
        raise ValueError(
            f"lin_cols must have shape ({n_aug},); got {tuple(lin_cols.shape)}"
        )
    B_full = np.zeros((n_in + 1, n_aug), dtype=np.float64)
    B_full[0, 0] = 1.0
    B_full[1:, 1:] = A
    B_masked = B_full * lin_cols[None, :]
    BBt = jnp.asarray(B_masked @ B_masked.T, dtype=X.dtype)
    Z = add_bias(X)
    return Z, BBt, Z @ BBt


def _evaluate_splits_kernel(
    Z: jnp.ndarray,  # (N, m)   raw bias-augmented inputs
    ZK: jnp.ndarray,  # (N, m)  Z @ (B Bᵀ)
    BBt: jnp.ndarray,  # (m, m) B Bᵀ
    y: jnp.ndarray,  # (N, T)
    node_mask: jnp.ndarray,  # (N,) float — 1 for in-subdomain
    X_split: jnp.ndarray,  # (N, K)
    thresholds: jnp.ndarray,  # (B, K)
    ridge: float,
    min_samples_leaf: float,
    loss_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate K*B candidate splits in parallel.

    Returns
    -------
    overall : (K, B) per-candidate combined loss (inf for invalid)
    n_below : (K, B) sample counts on the left
    n_above : (K, B) sample counts on the right

    The per-candidate leaf parameters are deliberately *not* returned:
    only the argmin candidate's children are ever needed, and they are
    refit host-side with :func:`_fit_one_side` (the same code path as
    leaf fits, so child/leaf parameters stay bit-identical). Materialising
    ``(K, B, n_aug, T)`` for every candidate was both the memory and the
    compile-time bottleneck on lifted feature spaces.
    """
    K = X_split.shape[1]
    B = thresholds.shape[0]
    N = Z.shape[0]

    # threshold_mask[c, b, s] = X_split[s, c] <= thresholds[b, c]
    # rearrange to (K, B, N)
    # X_split.T: (K, N); thresholds.T: (K, B)
    thr_below = X_split.T[:, None, :] <= thresholds.T[:, :, None]  # (K, B, N)
    # Combine with node mask. node_mask is (N,) float; cast threshold_mask to same dtype.
    nm = node_mask[None, None, :]
    mb = nm * thr_below.astype(node_mask.dtype)
    ma = nm * (1.0 - thr_below.astype(node_mask.dtype))

    n_below = jnp.sum(mb, axis=-1)
    n_above = jnp.sum(ma, axis=-1)
    valid = (n_below >= min_samples_leaf) & (n_above >= min_samples_leaf)

    # Flatten (K, B) → (K*B,) for vmap
    mb_flat = mb.reshape(K * B, N)
    ma_flat = ma.reshape(K * B, N)

    # Fitted values for each side of each candidate, computed in the
    # m-dimensional reduced space (design note 5).
    yh_below = _reduced_yhat_many(Z, ZK, BBt, y, mb_flat, ridge)  # (K*B, N, T)
    yh_above = _reduced_yhat_many(Z, ZK, BBt, y, ma_flat, ridge)

    weighted_loss = _weighted_loss(loss_name)
    # Evaluate the loss of the *combined* piecewise prediction over the
    # whole node (upstream convention). A row takes its own side's
    # prediction; rows outside the node get 0 and are excluded by the
    # node_mask weights. Criterion-correct for every loss: mean-type
    # losses (mae, mse, msle) reduce to the sample-weighted average of the two
    # sides, rmse pools under the sqrt, max_abs takes the max across sides.
    yh_combined = yh_below * mb_flat[:, :, None] + yh_above * ma_flat[:, :, None]
    overall_flat = jax.vmap(lambda yh: weighted_loss(y, yh, node_mask))(yh_combined)
    overall_flat = jnp.sum(overall_flat, axis=-1)  # sum over targets
    overall_flat = jnp.where(valid.reshape(K * B), overall_flat, jnp.inf)
    overall_flat = jnp.nan_to_num(overall_flat, nan=jnp.inf, posinf=jnp.inf)

    return overall_flat.reshape(K, B), n_below, n_above


# JIT-compile the kernel; static_argnames avoids retracing on hyperparams.
#
# Shape invariants (compile-once requires these to be stable across nodes):
# - Z.shape   == (N, m)                — m = n_features_in + 1
# - ZK.shape  == (N, m)
# - BBt.shape == (m, m)
# - y.shape   == (N, T)
# - node_mask.shape == (N,)
# - X_split.shape   == (N, K)          — K = chunk width, fixed at fit time
# - thresholds.shape == (B, K)         — B = max_bins - 1, set at fit time
# - ridge, min_samples_leaf are scalars; loss_name is a static string
# `K`, `B`, `N`, `m`, `T` are all fixed for the duration of a single
# `grow_tree` call, so this kernel JIT-traces once per fit (the last
# column chunk is padded to the same width rather than reshaped). If a
# future refactor makes split_features per-node-dynamic, retracing will
# fire on every node — guard the dynamic case explicitly there.
_evaluate_splits = jax.jit(
    _evaluate_splits_kernel,
    static_argnames=("loss_name",),
)


# Peak-intermediate budget for one call to `_evaluate_splits`. The split
# columns are processed in chunks narrow enough to stay under this, so a
# large lifted feature space costs more *calls*, not more memory. Raise
# it to trade memory for fewer kernel launches.
SPLIT_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024


def _candidate_chunk_cols(
    n_split_cols: int, n_bins: int, n_rows: int, n_targets: int, m: int
) -> int:
    """How many split columns fit in one kernel call under the budget.

    The kernel's dominant intermediates are ``O(C·N)`` side masks and
    fitted values (``C = cols·bins`` candidates) plus ``O(C·m²)`` Gram
    matrices. Solve for the largest ``cols`` whose working set fits, then
    clamp to ``[1, n_split_cols]``.
    """
    if n_bins <= 0:
        return max(1, n_split_cols)
    # bytes per candidate: masks (2), fitted values (3), plus Gram/rhs.
    per_candidate = 8 * (5 * n_rows * max(n_targets, 1) + m * m + 2 * m * n_targets)
    per_col = max(1, per_candidate * n_bins)
    cols = int(SPLIT_MEMORY_BUDGET_BYTES // per_col)
    return max(1, min(n_split_cols, cols))


def _evaluate_splits_chunked(
    Z: jnp.ndarray,
    ZK: jnp.ndarray,
    BBt: jnp.ndarray,
    y: jnp.ndarray,
    node_mask: jnp.ndarray,
    X_split: jnp.ndarray,
    thresholds: jnp.ndarray,
    ridge: float,
    min_samples_leaf: float,
    loss_name: str,
    chunk_cols: int,
) -> np.ndarray:
    """`_evaluate_splits` over column chunks; returns the (K, B) losses.

    Chunks are always ``chunk_cols`` wide — the tail is padded with
    ``+inf`` thresholds (every row lands below, so the right child is
    empty and the candidate scores ``inf``) and sliced off afterwards.
    Fixed width means one traced shape per fit, not one per chunk size.
    """
    K = int(X_split.shape[1])
    B = int(thresholds.shape[0])
    if chunk_cols >= K:
        overall, _, _ = _evaluate_splits(
            Z, ZK, BBt, y, node_mask, X_split, thresholds,
            ridge, min_samples_leaf, loss_name,
        )
        return np.asarray(overall)

    blocks: list[np.ndarray] = []
    for start in range(0, K, chunk_cols):
        stop = min(start + chunk_cols, K)
        xs = X_split[:, start:stop]
        th = thresholds[:, start:stop]
        pad = chunk_cols - (stop - start)
        if pad:
            xs = jnp.concatenate(
                [xs, jnp.zeros((xs.shape[0], pad), dtype=xs.dtype)], axis=1
            )
            th = jnp.concatenate(
                [th, jnp.full((B, pad), jnp.inf, dtype=th.dtype)], axis=1
            )
        overall, _, _ = _evaluate_splits(
            Z, ZK, BBt, y, node_mask, xs, th,
            ridge, min_samples_leaf, loss_name,
        )
        blocks.append(np.asarray(overall)[: stop - start])
    return np.concatenate(blocks, axis=0)


# Leaf-fit alias. Kept as the *eager* `_fit_one_side` rather than a
# separate `jax.jit(...)`: XLA is free to reorder the solve's
# floating-point ops, so a jitted leaf fit produces ULP-level different
# leaf coefficients than the eager path. That perturbation is harmless
# numerically but flips near-tied downstream consumers (notably discopt's
# big-M MIP, whose feasibility is numerically fragile). The leaf fit is a
# small fraction of fit cost (the split kernel dominates), so we keep the
# eager path for exact, reproducible leaf parameters.
_fit_leaf = _fit_one_side


# --------------------------------------------------------------------------
# Python-side: thresholds, leaf parameters, loss on subdomain
# --------------------------------------------------------------------------


def _compute_thresholds(
    X_split_subdomain,  # (N_active, K) — numpy or JAX-backed; converted to numpy
    max_bins: int,
) -> jnp.ndarray:
    """Quantile thresholds with the endpoint quantiles dropped.

    Returns ``(B, K)`` where ``B = max_bins - 1``, matching upstream's
    ``linspace(0,1,max_bins+1)[1:-1]`` convention.

    Computed via :func:`numpy.quantile` so the variable ``N_active`` size
    does not retrigger a JAX trace per node. The returned ``(B, K)``
    tensor is then sent to the JIT'd split kernel, which only retraces
    on its own static-shape inputs.
    """
    arr = np.asarray(X_split_subdomain)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    out = np.quantile(arr, qs, axis=0)
    return jnp.asarray(out)


@jax.jit
def _leaf_stats_kernel(
    mask: jnp.ndarray,
    X_t: jnp.ndarray,
    X_aug: jnp.ndarray,
    y: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-leaf summary statistics in a single jit'd kernel.

    Returns ``(x_mean, x_var, n, mse)``; the caller is free to convert
    ``n`` to a Python int once if it needs to.
    """
    nm = jnp.sum(mask)
    safe_nm = jnp.where(nm > 0.0, nm, 1.0)
    x_mean = jnp.sum(X_t * mask[:, None], axis=0) / safe_nm
    x_var = jnp.sum(((X_t - x_mean) ** 2) * mask[:, None], axis=0) / jnp.where(
        nm > 1.0, nm - 1.0, 1.0
    )
    yh = X_aug @ params
    sq = ((y - yh) ** 2) * mask[:, None]
    mse = jnp.sum(sq, axis=0) / safe_nm
    return x_mean, x_var, nm, mse


def _initial_leaf(
    X_aug: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    ridge: float,
    criterion: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a regression on the active rows and return (params, loss).

    The reported loss aggregates the *chosen* criterion across targets so
    it is comparable with the child-node losses tracked for
    ``min_impurity_decrease``.
    """
    params = _fit_leaf(X_aug, y, mask, ridge)
    yh = X_aug @ params
    fn = _WEIGHTED_LOSS_FNS[criterion]
    per_target = fn(y, yh, mask)  # (T,)
    loss = jnp.sum(per_target)
    return params, loss


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def grow_tree(
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    transform_matrix: Optional[jnp.ndarray] = None,
    linear_features: Optional[tuple[int, ...]] = None,
    split_features: Optional[tuple[int, ...]] = None,
    categorical_features: Optional[tuple[int, ...]] = None,
    criterion: str = "mae",
    max_depth: int = 32,
    max_bins: int = 10,
    min_samples_split: int = 6,
    min_samples_leaf: int | float = 0.01,
    min_impurity_decrease: float = 0.0,
    ridge: float = 1e-5,
    depth_first: bool = True,
) -> Tree:
    """Build a piecewise-linear decision tree.

    Low-level entry point used by :class:`LinearTreeRegressor` and
    :class:`HyperplaneTreeRegressor`. Most callers should prefer the
    regressor classes, which document each parameter at the public API
    level.

    Parameters
    ----------
    X : (N, n_features_in)
        Training inputs. Cast to float64 at the boundary.
    y : (N,) or (N, T)
        Training targets; ``T`` is the number of regression heads.
    transform_matrix : optional (n_features_in, n_features_transformed)
        If provided, the tree splits and regresses on
        ``X @ transform_matrix``. For axis-aligned LMDT, leave ``None``
        (the function inserts an identity).
    linear_features, split_features, categorical_features
        Index sets in the **transformed** column space; default to
        every non-categorical column.
    criterion : {"mae", "mse", "rmse", "msle", "max_abs"}
        Split-loss name. Callable criteria are not supported by the
        inner JIT kernel.
    max_depth, max_bins, min_samples_split, min_samples_leaf,
    min_impurity_decrease, ridge, depth_first
        See :class:`LinearTreeRegressor` for semantics.
    """
    # Force float64 at the boundary so a user passing float32 doesn't
    # silently downgrade the whole pipeline when JAX_ENABLE_X64=1. When
    # x64 is disabled JAX transparently demotes this back to float32.
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if y.ndim == 1:
        y = y[:, None]

    n_in = X.shape[1]
    if transform_matrix is None:
        transform_matrix = jnp.eye(n_in, dtype=X.dtype)
    n_transformed = transform_matrix.shape[1]

    X_t = X @ transform_matrix  # (N, n_transformed)
    X_aug = add_bias(X_t)  # (N, n_transformed + 1)
    n_aug = X_aug.shape[1]
    N, T = y.shape

    if categorical_features is None:
        categorical_features = ()
    if linear_features is None:
        linear_features = tuple(i for i in range(n_transformed) if i not in categorical_features)
    if split_features is None:
        split_features = tuple(i for i in range(n_transformed) if i not in categorical_features)

    # Restrict the leaf/split regressions to `linear_features` (+ bias).
    # Columns outside the set are zeroed in the *fitting* design matrix
    # `X_fit`; with a zeroed column, `XᵀWX` carries only `ridge` on that
    # diagonal and `Xᵀy = 0`, so the solved coefficient is exactly 0.
    # `X_aug` (all columns) is still used for prediction/UQ downstream,
    # where those zero coefficients are inert — so `leaf_params` keeps
    # its `(n_aug, T)` shape and predict/ONNX/JSON/discopt are unchanged.
    # This is what makes `linear_features` (and, by default, categorical
    # exclusion) actually affect the leaf models. Routing still uses the
    # raw `X_t` columns, so `split_features` is unaffected.
    lin_cols = np.zeros(n_aug, dtype=bool)
    lin_cols[0] = True  # bias term always retained
    for f in linear_features:
        lin_cols[int(f) + 1] = True
    X_fit = X_aug * jnp.asarray(lin_cols, dtype=X_aug.dtype)[None, :]

    # Reduced basis for candidate-split evaluation (design note 5).
    m_reduced = n_in + 1
    Z_raw, BBt, ZK = reduced_split_basis(X, transform_matrix, lin_cols)

    # Resolve min_samples_leaf to absolute integer.
    if isinstance(min_samples_leaf, float) and 0.0 < min_samples_leaf < 1.0:
        msl = max(3, int(jnp.ceil(min_samples_leaf * N)))
    else:
        msl = max(3, int(min_samples_leaf))

    if isinstance(min_samples_split, float) and 0.0 < min_samples_split < 1.0:
        mss = max(2 * msl, int(jnp.ceil(min_samples_split * N)))
    else:
        mss = max(2 * msl, int(min_samples_split))

    # The inner JIT kernel selects a weighted-loss closure by name (it
    # uses a Python `dict` keyed on `loss_name`, not a callable). Custom
    # callable criteria therefore cannot drive split selection without
    # silently falling through to MAE — we reject them here so users see
    # the limitation up front instead of getting wrong-criterion splits.
    if not isinstance(criterion, str):
        raise TypeError(
            "tree growth requires a string criterion (one of "
            f"{sorted(_WEIGHTED_LOSS_FNS)}); callable criteria are not "
            "supported in the inner kernel."
        )
    if criterion not in _WEIGHTED_LOSS_FNS:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Pick one of "
            f"{sorted(_WEIGHTED_LOSS_FNS)}."
        )

    # --- root node ---
    root_mask = jnp.ones(N, dtype=X_aug.dtype)

    # node bookkeeping
    nodes: list[_GrowNode] = []
    node_masks: list[jnp.ndarray] = []
    node_loss: list[float] = []

    def _leaf_stats(
        mask: jnp.ndarray, params: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute (x_mean, x_var, n, mse) for the active rows under ``mask``.

        Computed in a single jit'd callable so the four reductions
        (``nm``, ``x_mean``, ``x_var``, ``mse``) share one device kernel
        invocation. ``nm`` is returned as a *device* scalar (not
        ``int(nm)``): a per-leaf ``int()`` would force a host sync inside
        the growth loop. The scalars are stacked and converted to int32
        once at the final packing step instead.
        """
        return _leaf_stats_kernel(mask, X_t, X_aug, y, params)

    def _finalize_leaf(node_id: int, mask: jnp.ndarray) -> None:
        """Mark an existing placeholder node as a leaf and populate its
        UQ stats. Used by every early-exit path in the growth loop.

        The node's regression was already fitted when the node was
        created (`_add_node`), so no refit happens here — a node's
        parameters are solved exactly once, whether it ends up a leaf or
        an internal node.
        """
        nd = nodes[node_id]
        x_mean, x_var, n_leaf, mse = _leaf_stats(mask, nd.leaf_params)
        nd.is_leaf = True
        nd.leaf_x_mean = x_mean
        nd.leaf_x_var = x_var
        nd.leaf_n = n_leaf
        nd.leaf_mse = mse

    def _add_node(mask: jnp.ndarray) -> int:
        """Fit the node's regression, register it, and record its loss.

        Returns the new node id. The node starts as a non-leaf
        placeholder; `_finalize_leaf` promotes it if growth stops here.
        """
        params, loss = _initial_leaf(X_fit, y, mask, ridge, criterion)
        nid = len(nodes)
        nodes.append(
            _GrowNode(
                node_id=nid,
                is_leaf=False,
                feature=-1,
                threshold=float("nan"),
                left=-1,
                right=-1,
                leaf_params=params,
                leaf_x_mean=jnp.zeros((n_transformed,), dtype=X_aug.dtype),
                leaf_x_var=jnp.zeros((n_transformed,), dtype=X_aug.dtype),
                leaf_n=0,
                leaf_mse=jnp.zeros((T,), dtype=X_aug.dtype),
            )
        )
        node_masks.append(mask)
        node_loss.append(float(loss))
        return nid

    # --- queue: list of (node_id, depth) ---
    root_id = _add_node(root_mask)
    queue: list[tuple[int, int]] = [(root_id, 0)]

    split_feats_arr = jnp.asarray(split_features, dtype=jnp.int32)
    # Loop-invariant: the split-feature columns of X_t are the same for
    # every node (only the mask varies), so gather them once.
    X_split = X_t[:, split_feats_arr]

    # Column-chunk width for the candidate kernel. Fixed for the whole
    # call so the kernel traces once; see `_candidate_chunk_cols`.
    chunk_cols = _candidate_chunk_cols(
        int(X_split.shape[1]), max(max_bins - 1, 1), N, T, m_reduced
    )

    # Numpy mirror of `X_t` used for all host-side mask arithmetic.
    # Computed once per `grow_tree` call. The JIT'd split kernel still
    # consumes the JAX `X_t`; only the orchestration moves to numpy.
    X_t_np = np.asarray(X_t)

    while queue:
        idx = -1 if depth_first else 0
        node_id, depth = queue.pop(idx)
        mask = node_masks[node_id]
        # One bulk transfer per iteration (instead of multiple scalar
        # device→host reads scattered through the body).
        mask_np = np.asarray(mask)
        n_active = int(mask_np.sum())

        # Stopping criteria
        if (
            depth >= max_depth
            or n_active < mss
            or n_active < 2 * msl
        ):
            _finalize_leaf(node_id, mask)
            continue

        # Compute thresholds on the active subdomain via numpy (avoids
        # a per-node JAX retrace; the kernel call below is what does the
        # real work, and it has fixed shapes).
        active_idx = np.flatnonzero(mask_np > 0.0)
        X_split_subdomain = X_t_np[np.ix_(active_idx, split_features)]
        thresholds = _compute_thresholds(X_split_subdomain, max_bins)
        if thresholds.shape[0] == 0:
            _finalize_leaf(node_id, mask)
            continue

        # Candidate losses come back on the host already (the kernel is
        # called once per column chunk); all selection happens on numpy.
        overall_np = _evaluate_splits_chunked(
            Z_raw,
            ZK,
            BBt,
            y,
            mask,
            X_split,
            thresholds,
            float(ridge),
            float(msl),
            criterion,
            chunk_cols,
        )
        if not np.isfinite(overall_np).any():
            _finalize_leaf(node_id, mask)
            continue

        flat_idx = int(np.argmin(overall_np))
        B = overall_np.shape[1]
        c_idx, b_idx = divmod(flat_idx, B)
        best_loss = float(overall_np[c_idx, b_idx])

        # Stop if the best candidate split does not improve loss by at
        # least `min_impurity_decrease`. Default 0.0 means "any
        # non-negative improvement is accepted"; positive values prune.
        impurity_decrease = node_loss[node_id] - best_loss
        if impurity_decrease < min_impurity_decrease:
            _finalize_leaf(node_id, mask)
            continue

        split_feature_in_transformed_space = int(split_features[c_idx])
        split_threshold = float(np.asarray(thresholds)[b_idx, c_idx])

        # Build child masks on host then promote back to JAX for kernel
        # consumption downstream (and for the `_leaf_stats` helper).
        below_np = X_t_np[:, split_feature_in_transformed_space] <= split_threshold
        mask_below_np = mask_np * below_np.astype(mask_np.dtype)
        mask_above_np = mask_np * (1.0 - below_np.astype(mask_np.dtype))
        mask_below = jnp.asarray(mask_below_np)
        mask_above = jnp.asarray(mask_above_np)

        # Create the child nodes. `_add_node` fits each child's
        # regression and records its loss under the *same* criterion used
        # for split selection, so min_impurity_decrease comparisons on
        # grandchildren are consistent — and the fit is reused verbatim
        # if the child ends up a leaf.
        left_id = _add_node(mask_below)
        right_id = _add_node(mask_above)

        nodes[node_id].is_leaf = False
        nodes[node_id].feature = split_feature_in_transformed_space
        nodes[node_id].threshold = split_threshold
        nodes[node_id].left = left_id
        nodes[node_id].right = right_id

        queue.append((left_id, depth + 1))
        queue.append((right_id, depth + 1))

    # Every leaf is finalised inline via `_finalize_leaf` at its
    # early-exit path, so no post-loop re-fit is needed.

    # Pack into Tree
    n_nodes = len(nodes)
    is_leaf = jnp.array([n.is_leaf for n in nodes], dtype=jnp.bool_)
    feature = jnp.array([n.feature for n in nodes], dtype=jnp.int32)
    threshold = jnp.array([n.threshold for n in nodes], dtype=X_aug.dtype)
    left = jnp.array([n.left for n in nodes], dtype=jnp.int32)
    right = jnp.array([n.right for n in nodes], dtype=jnp.int32)
    leaf_params = jnp.stack([n.leaf_params for n in nodes], axis=0)  # (n_nodes, n_aug, T)

    leaf_uq = LeafUQ(
        n=jnp.array([n.leaf_n for n in nodes], dtype=jnp.int32),
        x_mean=jnp.stack([n.leaf_x_mean for n in nodes], axis=0),
        x_var=jnp.stack([n.leaf_x_var for n in nodes], axis=0),
        mse=jnp.stack([n.leaf_mse for n in nodes], axis=0),
    )

    return Tree(
        is_leaf=is_leaf,
        feature=feature,
        threshold=threshold,
        left=left,
        right=right,
        leaf_params=leaf_params,
        transform_matrix=transform_matrix,
        leaf_uq=leaf_uq,
        n_features_in=int(n_in),
        n_features_transformed=int(n_transformed),
        n_targets=int(T),
        linear_features=tuple(linear_features),
        categorical_features=tuple(categorical_features),
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def _route_one(tree: Tree, x_t: jnp.ndarray) -> jnp.ndarray:
    """Route a single (transformed) sample through the tree, returning leaf id.

    Uses lax.while_loop so it works under jit / vmap.
    """
    def cond(state: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        node_id, _done = state
        return ~tree.is_leaf[node_id]

    def body(state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        node_id, _ = state
        feat = tree.feature[node_id]
        thr = tree.threshold[node_id]
        go_left = x_t[feat] <= thr
        nxt = jnp.where(go_left, tree.left[node_id], tree.right[node_id])
        return nxt, jnp.array(False)

    init = (jnp.int32(0), jnp.array(False))
    final_id, _ = jax.lax.while_loop(cond, body, init)
    return final_id


@jax.jit
def _predict_jit(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """JIT'd core prediction returning (N, T). `Tree` is a registered
    pytree (its shape metadata is static aux), so this traces once per
    fitted tree and reuses the compiled kernel across same-shape calls —
    the many-small-calls pattern in acquisition/AL loops in particular."""
    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)
    leaf_ids = jax.vmap(lambda x: _route_one(tree, x))(X_t)  # (N,)
    params_per_sample = tree.leaf_params[leaf_ids]  # (N, n_aug, T)
    return jnp.einsum("nf,nft->nt", X_aug, params_per_sample)


def predict(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """Predict for X (N, n_features_in). Returns (N,) or (N, T)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    yh = _predict_jit(tree, X)
    if tree.n_targets == 1:
        yh = yh[:, 0]
    return yh


@jax.jit
def _apply_jit(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    X_t = X @ tree.transform_matrix
    return jax.vmap(lambda x: _route_one(tree, x))(X_t)


def apply_tree(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """Return leaf node id for each input row."""
    X = jnp.asarray(X, dtype=jnp.float64)
    return _apply_jit(tree, X)


# Backwards-compatible alias for the previous private name.
apply_ = apply_tree
