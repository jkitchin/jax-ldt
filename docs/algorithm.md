# Algorithm notes

This page summarises the math for what the trees actually compute. For
the user-facing API, see [the quickstart](quickstart.md).

## Tree structure

A trained tree is a flat parallel-array layout (see `jax_ldt._types.Tree`):

| field              | shape                          | meaning                              |
|--------------------|--------------------------------|--------------------------------------|
| `is_leaf`          | (n_nodes,)                     | leaf marker                          |
| `feature`          | (n_nodes,)                     | split feature in *transformed* space |
| `threshold`        | (n_nodes,)                     | split threshold; NaN at leaves       |
| `left`, `right`    | (n_nodes,)                     | child node ids                       |
| `leaf_params`      | (n_nodes, n_aug, n_targets)    | `[bias; weights]` per leaf           |
| `transform_matrix` | (n_features_in, n_lifted)      | maps X to lifted features            |

For a sample `x`:
1. **Lift**: `x' = x @ A` where `A = transform_matrix`.
2. **Augment**: `x_aug = [1; x']`.
3. **Route**: starting at node 0, follow `left` if `x'[feature] ≤
   threshold`, else `right`, until a leaf.
4. **Predict**: `ŷ = x_aug · leaf_params[leaf_id]`.

For axis-aligned LMDTs, `A` is the identity. For HTs, the first
`n_features_in` columns of `A` are the identity (preserving original
features) and the rest encode oblique directions.

## Hyperplane enumeration

`hyperplanes.build_transform_matrix` constructs `A` in three steps:

1. **Generate integer-weight rows** via Miller-index enumeration. For
   `(num_terms=2, max_weight=W)`, this yields all unique rows
   `[1, w]` with `w ∈ {0, 1/W, …, W}`.
2. **Symmetrise**: apply ±-parity and column permutations, prune rows
   with non-trailing zeros, normalise to leftmost-1, dedup at
   `tol_decimals` precision.
3. **Permute over input features**: for each ordered length-`num_terms`
   selection of non-categorical input columns, place each LC row at
   those positions.
4. **Optional scaling**: divide each row by feature range so oblique
   angles aren't dominated by feature units; renormalise.

The result is appended to an identity block so axis-aligned splits are
still candidates.

## Recursive split-and-grow

Tree growth is a Python while-loop over a queue of (node_id, depth).
Per node:

1. Compute candidate thresholds: `max_bins - 1` quantile cuts of each
   split feature on the active subdomain.
2. **Inner kernel** (JIT-compiled): for each candidate (column, bin),
   fit two ridge regressions (left and right side) using the side mask
   as a 0/1 sample weight, then evaluate weighted MAE / RMSE. The
   K·B candidates are vmapped — a single fused kernel handles them.
3. Pick the candidate with lowest combined loss; if no valid split or
   the depth limit is reached, finalise the node as a leaf.

The split kernel is shape-stable (sample axis is always `N`, the full
training set) so JIT compiles once. Subdomain selection is via mask
multiplication, not slicing.

## Why ridge regression?

We solve `(XᵀWX + ridge·I) β = XᵀWy` because:
- the ridge regularisation guarantees positive-definiteness of `XᵀWX`,
  so `jax.scipy.linalg.solve(..., assume_a='pos')` is faster and more
  stable than `lstsq`;
- it's well-defined even when `W` has very few nonzero entries — important
  because invalid candidate splits still produce *some* coefficients
  during the vmapped batch (we discard them via a validity mask after).

The order of ridge application is fixed: ridge is added to `XᵀWX`
*before* the solve, never after. This is pinned by
`tests/test_principled_choices.py` (planned follow-up) but is also
documented in `linear_regression.py`.
