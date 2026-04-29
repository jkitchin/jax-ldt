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

3. For each candidate split (col, bin), we fit two ridge regressions
   (left and right side) using the mask × side-mask as a 0/1
   sample-weight. This is mathematically equivalent to row-selection
   followed by ordinary ridge — but the shape stays (N, n_aug) so vmap
   covers all K*B candidates at once.

4. The fit uses normal-equations + `jax.scipy.linalg.solve(..., 'pos')`.
   Ridge is added to XᵀWX BEFORE the solve. Ridge guarantees positive
   definiteness even when w is sparse / nearly all-zero, so invalid
   candidates produce some theta which we then discard via a validity
   mask. See `test_principled_choices.py`.
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
    leaf_n: int
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


def _evaluate_splits_kernel(
    X_aug: jnp.ndarray,  # (N, n_aug)
    y: jnp.ndarray,  # (N, T)
    node_mask: jnp.ndarray,  # (N,) float — 1 for in-subdomain
    X_split: jnp.ndarray,  # (N, K)
    thresholds: jnp.ndarray,  # (B, K)
    ridge: float,
    min_samples_leaf: float,
    loss_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate K*B candidate splits in parallel.

    Returns
    -------
    overall : (K, B) per-candidate combined loss (inf for invalid)
    theta_below : (K, B, n_aug, T)
    theta_above : (K, B, n_aug, T)
    n_below : (K, B) sample counts on the left
    n_above : (K, B) sample counts on the right
    """
    K = X_split.shape[1]
    B = thresholds.shape[0]
    N = X_aug.shape[0]

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

    theta_below_flat = _fit_many_sides(X_aug, y, mb_flat, ridge)  # (K*B, n_aug, T)
    theta_above_flat = _fit_many_sides(X_aug, y, ma_flat, ridge)

    # Predict for all N rows under each candidate
    # X_aug @ theta gives (N, T); for K*B candidates we einsum
    yh_below = jnp.einsum("nf,ift->int", X_aug, theta_below_flat)  # (K*B, N, T)
    yh_above = jnp.einsum("nf,ift->int", X_aug, theta_above_flat)

    weighted_loss = _weighted_loss(loss_name)
    loss_below_flat = jax.vmap(lambda yh, w: weighted_loss(y, yh, w))(yh_below, mb_flat)
    loss_above_flat = jax.vmap(lambda yh, w: weighted_loss(y, yh, w))(yh_above, ma_flat)

    n_total = jnp.sum(node_mask)
    safe_n_total = jnp.where(n_total > 0.0, n_total, 1.0)
    overall_flat = (
        loss_below_flat * n_below.reshape(K * B, 1)
        + loss_above_flat * n_above.reshape(K * B, 1)
    ) / safe_n_total
    overall_flat = jnp.sum(overall_flat, axis=-1)  # sum over targets
    overall_flat = jnp.where(valid.reshape(K * B), overall_flat, jnp.inf)
    overall_flat = jnp.nan_to_num(overall_flat, nan=jnp.inf, posinf=jnp.inf)

    return (
        overall_flat.reshape(K, B),
        theta_below_flat.reshape(K, B, X_aug.shape[1], y.shape[1]),
        theta_above_flat.reshape(K, B, X_aug.shape[1], y.shape[1]),
        n_below,
        n_above,
    )


# JIT-compile the kernel; static_argnames avoids retracing on hyperparams.
#
# Shape invariants (compile-once requires these to be stable across nodes):
# - X_aug.shape  == (N, n_aug)         — full dataset, augmented bias column
# - y.shape      == (N, T)
# - node_mask.shape == (N,)
# - X_split.shape   == (N, K)          — K = len(split_features), set at fit time
# - thresholds.shape == (B, K)         — B = max_bins - 1, set at fit time
# - ridge, min_samples_leaf are scalars; loss_name is a static string
# `K`, `B`, `N`, `n_aug`, `T` are all fixed for the duration of a single
# `grow_tree` call, so this kernel JIT-traces once per fit. If a future
# refactor makes split_features per-node-dynamic, retracing will fire on
# every node — guard the dynamic case explicitly there.
_evaluate_splits = jax.jit(
    _evaluate_splits_kernel,
    static_argnames=("loss_name",),
)


# Leaf-fit alias: the JIT cache from `_fit_one_side` (used vmapped during
# growth) and a direct call use the same primitives, so we just expose the
# function under a clearer name. Removing the previous extra `jax.jit(...)`
# wrapper avoids a redundant trace at leaf-finalisation time.
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
    criterion : {"mae", "rmse", "msle", "max_abs"}
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
    root_params, root_loss = _initial_leaf(X_aug, y, root_mask, ridge, criterion)

    # node bookkeeping
    nodes: list[_GrowNode] = []
    node_masks: list[jnp.ndarray] = []
    node_loss: list[float] = []

    def _leaf_stats(
        mask: jnp.ndarray, params: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
        """Compute (x_mean, x_var, n, mse) for the active rows under ``mask``.

        Computed in a single jit'd callable so the four reductions
        (``nm``, ``x_mean``, ``x_var``, ``mse``) share one device kernel
        invocation. The previous version did each reduction separately
        and triggered a host sync inside the loop via ``int(nm)``; we
        return ``nm`` as a JAX scalar and convert at the call site.
        """
        x_mean, x_var, nm, mse = _leaf_stats_kernel(mask, X_t, X_aug, y, params)
        return x_mean, x_var, int(nm), mse

    def _finalize_leaf(node_id: int, mask: jnp.ndarray, params: jnp.ndarray) -> None:
        """Mark an existing placeholder node as a leaf and populate its
        UQ stats. Used by every early-exit path in the growth loop."""
        x_mean, x_var, n_leaf, mse = _leaf_stats(mask, params)
        nd = nodes[node_id]
        nd.is_leaf = True
        nd.leaf_params = params
        nd.leaf_x_mean = x_mean
        nd.leaf_x_var = x_var
        nd.leaf_n = n_leaf
        nd.leaf_mse = mse

    def _placeholder_split(mask: jnp.ndarray) -> int:
        nid = len(nodes)
        nodes.append(
            _GrowNode(
                node_id=nid,
                is_leaf=False,
                feature=-1,
                threshold=float("nan"),
                left=-1,
                right=-1,
                leaf_params=jnp.zeros((n_aug, T), dtype=X_aug.dtype),
                leaf_x_mean=jnp.zeros((n_transformed,), dtype=X_aug.dtype),
                leaf_x_var=jnp.zeros((n_transformed,), dtype=X_aug.dtype),
                leaf_n=0,
                leaf_mse=jnp.zeros((T,), dtype=X_aug.dtype),
            )
        )
        node_masks.append(mask)
        return nid

    # --- queue: list of (node_id, depth) ---
    root_id = _placeholder_split(root_mask)
    node_loss.append(float(root_loss))
    queue: list[tuple[int, int]] = [(root_id, 0)]

    split_feats_arr = jnp.asarray(split_features, dtype=jnp.int32)

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
            params, _ = _initial_leaf(X_aug, y, mask, ridge, criterion)
            _finalize_leaf(node_id, mask, params)
            continue

        # Compute thresholds on the active subdomain via numpy (avoids
        # a per-node JAX retrace; the kernel call below is what does the
        # real work, and it has fixed shapes).
        active_idx = np.flatnonzero(mask_np > 0.0)
        X_split_subdomain = X_t_np[np.ix_(active_idx, split_features)]
        thresholds = _compute_thresholds(X_split_subdomain, max_bins)
        if thresholds.shape[0] == 0:
            params, _ = _initial_leaf(X_aug, y, mask, ridge, criterion)
            _finalize_leaf(node_id, mask, params)
            continue

        X_split = X_t[:, split_feats_arr]
        overall, theta_b, theta_a, n_below, n_above = _evaluate_splits(
            X_aug,
            y,
            mask,
            X_split,
            thresholds,
            float(ridge),
            float(msl),
            criterion,
        )

        # Pull the kernel outputs to host once and do all selection on
        # numpy. Reading individual scalars from a JAX array would
        # trigger a separate transfer per access.
        overall_np = np.asarray(overall)
        if not np.isfinite(overall_np).any():
            params, _ = _initial_leaf(X_aug, y, mask, ridge, criterion)
            _finalize_leaf(node_id, mask, params)
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
            params, _ = _initial_leaf(X_aug, y, mask, ridge, criterion)
            _finalize_leaf(node_id, mask, params)
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

        # Reserve child nodes. Track child loss under the *same* criterion
        # used for split selection so min_impurity_decrease comparisons
        # on grandchildren are consistent.
        left_id = _placeholder_split(mask_below)
        node_loss.append(
            float(_weighted_loss_python(criterion, y, X_aug @ theta_b[c_idx, b_idx], mask_below))
        )
        right_id = _placeholder_split(mask_above)
        node_loss.append(
            float(_weighted_loss_python(criterion, y, X_aug @ theta_a[c_idx, b_idx], mask_above))
        )

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


def _weighted_loss_python(
    name: str, y: jnp.ndarray, yh: jnp.ndarray, w: jnp.ndarray
) -> float:
    """Evaluate a weighted loss under the chosen criterion.

    Used for tracking child-node loss for ``min_impurity_decrease``;
    must agree with the criterion driving the split-selection kernel.
    Returns inf for an empty subdomain so a downstream "improvement"
    check rejects it.
    """
    sw = float(jnp.sum(w))
    if sw <= 0.0:
        return float("inf")
    fn = _WEIGHTED_LOSS_FNS[name]
    out = fn(y, yh, w)
    return float(jnp.sum(out))


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


def predict(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """Predict for X (N, n_features_in). Returns (N,) or (N, T)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)

    leaf_ids = jax.vmap(lambda x: _route_one(tree, x))(X_t)  # (N,)
    # Gather leaf params per sample: (N, n_aug, T)
    params_per_sample = tree.leaf_params[leaf_ids]
    # einsum to multiply
    yh = jnp.einsum("nf,nft->nt", X_aug, params_per_sample)
    if tree.n_targets == 1:
        yh = yh[:, 0]
    return yh


def apply_tree(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """Return leaf node id for each input row."""
    X = jnp.asarray(X, dtype=jnp.float64)
    X_t = X @ tree.transform_matrix
    return jax.vmap(lambda x: _route_one(tree, x))(X_t)


# Backwards-compatible alias for the previous private name.
apply_ = apply_tree
