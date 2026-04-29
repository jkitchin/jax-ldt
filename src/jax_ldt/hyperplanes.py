"""Hyperplane (oblique-split) feature lifting.

Ported from upstream `linear_combinations.py`. Generates a finite set of
linear combinations of input features (hyperplane weights), then builds
a transformation matrix `A` of shape `(n_features_in, n_features_lifted)`
where the first `n_features_in` columns are the identity (preserves
original axis-aligned splits) and the rest encode each unique
oblique direction.

The combinatorial enumeration uses Python `itertools` (it runs once at
fit time and the output is small); the matrix building is plain JAX.

Convention: weight enumeration is deterministic and lex-sorted after
rounding to `tol_decimals`. This is pinned by `test_principled_choices.py`.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from typing import Optional

import jax.numpy as jnp
import numpy as np


def _warn_aggressive_dedup(before: int, after: int, tol_decimals: int, where: str) -> None:
    """Warn if more than 10% of rows were dropped in dedup-by-rounding.

    Aggressive rounding (low ``tol_decimals``) silently collapses
    near-duplicate hyperplane rows. When the drop rate is large, the
    user is most likely losing distinct directions; surface a warning
    so they can raise ``tol_decimals``.
    """
    if before <= 0:
        return
    dropped = before - after
    if dropped <= 0:
        return
    if dropped / before > 0.10:
        warnings.warn(
            f"hyperplanes.{where}: rounded.dedup dropped {dropped} of {before} rows "
            f"(kept {after}) at tol_decimals={tol_decimals}. "
            f"Consider increasing tol_decimals to preserve distinct directions.",
            UserWarning,
            stacklevel=3,
        )


@functools.lru_cache(maxsize=64)
def generate_planes_to_index(
    dimension: int,
    max_weight: int = 3,
    tol_decimals: int = 4,
) -> tuple[tuple[float, ...], ...]:
    """Miller-index-like enumeration of integer-weight hyperplanes.

    Returns a tuple of tuples (immutable, hashable, lru_cache-friendly).
    Each row has its first element normalized to 1; rows are deduplicated
    after rounding to `tol_decimals`.
    """
    if max_weight == 0:
        # special case: identity only
        return tuple(tuple(row) for row in np.eye(dimension).tolist())

    combos = list(
        itertools.combinations_with_replacement(range(max_weight, -1, -1), dimension)
    )
    arr = np.asarray(combos, dtype=np.float64)
    if len(arr) == 0:
        return ()
    # Drop the all-zero row explicitly (cannot rely on the iteration
    # order — `arr[:-1]` would silently keep an unintended row if the
    # ordering of `combinations_with_replacement` ever changed).
    arr = arr[~np.all(arr == 0.0, axis=1)]
    if arr.shape[0] == 0:
        return ()
    # normalize: divide by first element
    first = arr[:, 0:1]
    # avoid div-by-zero if first elt is zero (shouldn't happen since combos are sorted desc)
    first = np.where(first == 0, 1.0, first)
    arr = arr / first
    arr = np.round(arr, decimals=tol_decimals)
    # unique + sorted
    arr = np.unique(arr, axis=0)
    return tuple(tuple(row.tolist()) for row in arr)


def symmetrize(
    LCs: jnp.ndarray,
    tol_decimals: int = 4,
) -> jnp.ndarray:
    """Apply ±-parity and permutation symmetries to a hyperplane matrix.

    Input: (n_lcs, num_terms). Output: (n_unique, num_terms).
    """
    LCs = jnp.asarray(LCs, dtype=jnp.float64)
    num_terms = int(LCs.shape[1])

    # ±-parity: multiply by every (±1, ±1, ..., ±1)
    parity = jnp.asarray(list(itertools.product([1.0, -1.0], repeat=num_terms)))
    # (n_parity, 1, num_terms) * (1, n_lcs, num_terms) → (n_parity, n_lcs, num_terms)
    expanded = parity[:, None, :] * LCs[None, :, :]
    LCs = expanded.reshape(-1, num_terms)

    # column permutations
    perms = jnp.asarray(list(itertools.permutations(range(num_terms))), dtype=jnp.int32)
    # (n_lcs, n_perms, num_terms)
    permuted = LCs[:, perms]
    LCs = permuted.reshape(-1, num_terms)

    # remove rows with non-trailing zeros: a row like (0, 1) is invalid;
    # (1, 0) is valid. We keep rows where if any component is zero, all
    # subsequent components must also be zero — equivalently, the
    # nonzero indicators are monotonically non-increasing.
    nonzero = (LCs != 0).astype(jnp.int32)
    # cum-min from the left across columns: once a zero shows up, must stay zero.
    # Equivalent check: nonzero[:, i] <= nonzero[:, i-1] for all i>0.
    # We'll just compute it numpy-side since this is shape-static.
    nz_np = np.asarray(nonzero)
    keep = np.ones(nz_np.shape[0], dtype=bool)
    for i in range(1, num_terms):
        keep &= nz_np[:, i] <= nz_np[:, i - 1]
    LCs = LCs[jnp.asarray(keep)]

    # normalize: divide by first element
    first = LCs[:, 0:1]
    first = jnp.where(first == 0, 1.0, first)
    LCs = LCs / first

    # round + unique (numpy-side for stable lex sort)
    rounded = np.round(np.asarray(LCs), decimals=tol_decimals)
    uniq = np.unique(rounded, axis=0)
    _warn_aggressive_dedup(rounded.shape[0], uniq.shape[0], tol_decimals, "symmetrize")
    return jnp.asarray(uniq, dtype=jnp.float64)


def _factorial(n: int) -> int:
    return math.factorial(n)


def build_transform_matrix(
    X: jnp.ndarray,
    *,
    num_terms: int = 2,
    max_weight: int = 1,
    LCs: Optional[jnp.ndarray] = None,
    do_symmetrize: bool = True,
    do_scaling: bool = True,
    tol_decimals: int = 4,
    categorical_features: Optional[tuple[int, ...]] = None,
) -> jnp.ndarray:
    """Build the (n_features_in, n_features_lifted) transformation matrix.

    The first n_features_in columns are the identity (preserve original
    features). The rest encode unique oblique directions formed by
    placing each LC row at every length-`num_terms` permutation of the
    non-categorical column indices.

    `do_scaling` rescales each LC row by the data range of each feature
    (so hyperplane angles aren't dominated by feature units).

    Returned matrix shape: (n_in, n_lifted). To lift X: `X @ A`.
    """
    X = jnp.asarray(X)
    n_cols = int(X.shape[1])

    if categorical_features is None:
        categorical_features = ()
    cat_set = set(categorical_features)
    cols_to_perm = tuple(c for c in range(n_cols) if c not in cat_set)
    n_non_cat = len(cols_to_perm)

    if num_terms > n_non_cat:
        raise ValueError(
            f"num_terms={num_terms} exceeds non-categorical feature count {n_non_cat}"
        )

    if LCs is None:
        rows = generate_planes_to_index(num_terms, max_weight=max_weight, tol_decimals=tol_decimals)
        LCs_arr = jnp.asarray(rows, dtype=jnp.float64)
    else:
        LCs_arr = jnp.asarray(LCs, dtype=jnp.float64)

    if do_symmetrize:
        LCs_arr = symmetrize(LCs_arr, tol_decimals=tol_decimals)

    # Drop rows that are pure-axis-aligned (only one nonzero == 1) — those
    # would duplicate the identity block we prepend below.
    abs_sums = jnp.sum(jnp.abs(LCs_arr), axis=1)
    keep = abs_sums != 1.0
    LCs_arr = LCs_arr[keep]
    n_lcs = int(LCs_arr.shape[0])

    if n_lcs == 0:
        return jnp.eye(n_cols, dtype=X.dtype)

    perms_arr = np.asarray(
        list(itertools.permutations(cols_to_perm, num_terms)), dtype=np.int64
    )  # (n_perms, num_terms)
    n_perms = perms_arr.shape[0]
    n_extra = n_perms * n_lcs

    # Vectorised scatter: for each (permutation p, LC row j) place
    # ``LCs[j, k]`` at column ``perms_arr[p, k]`` of row ``p*n_lcs + j``.
    # The previous implementation did this with two nested Python loops,
    # which became the dominant fit-time cost for large `num_terms` /
    # ``n_non_cat`` (e.g. 380×n_lcs rows for num_terms=2, n_non_cat=20).
    LCs_np = np.asarray(LCs_arr)
    extra = np.zeros((n_extra, n_cols), dtype=np.float64)
    row_idx = np.repeat(np.arange(n_extra), num_terms)
    col_idx = np.broadcast_to(perms_arr[:, None, :], (n_perms, n_lcs, num_terms)).reshape(-1)
    vals = np.broadcast_to(LCs_np[None, :, :], (n_perms, n_lcs, num_terms)).reshape(-1)
    extra[row_idx, col_idx] = vals

    # Normalize each row by its leftmost nonzero entry.
    leftmost_idx = np.argmax(extra != 0, axis=1)
    leftmost_val = extra[np.arange(extra.shape[0]), leftmost_idx]
    leftmost_val = np.where(leftmost_val == 0, 1.0, leftmost_val)
    extra = extra / leftmost_val[:, None]

    # Round + unique to drop duplicates introduced by permutations.
    extra_before = extra.shape[0]
    extra = np.round(extra, decimals=tol_decimals)
    extra = np.unique(extra, axis=0)
    _warn_aggressive_dedup(
        extra_before, extra.shape[0], tol_decimals, "build_transform_matrix"
    )

    if do_scaling:
        ranges = np.asarray(jnp.max(X, axis=0) - jnp.min(X, axis=0), dtype=np.float64)
        ranges = np.where(ranges < 1e-8, 1e-8, ranges)
        extra = extra / ranges[None, :]
        leftmost_idx = np.argmax(extra != 0, axis=1)
        leftmost_val = extra[np.arange(extra.shape[0]), leftmost_idx]
        leftmost_val = np.where(leftmost_val == 0, 1.0, leftmost_val)
        extra = extra / leftmost_val[:, None]

    # Stack identity then extra: total shape (n_cols + n_extra_unique, n_cols).
    # Then transpose to (n_cols, n_total) so X @ A gives lifted features.
    full = np.vstack([np.eye(n_cols), extra]).astype(np.float64)
    A = jnp.asarray(full.T, dtype=X.dtype)
    return A


def lift(X: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    """Apply the transformation: `X @ A`."""
    return jnp.asarray(X) @ A
