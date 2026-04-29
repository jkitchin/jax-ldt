"""Adapter from jax-ldt Tree → discopt Model embedding.

`embed_in_discopt_model(model, m, x_vars)` augments the user's
`discopt.Model` with the variables and constraints needed to express
the trained tree's output as a discopt expression that can be referenced
in objectives or other constraints.

What's encoded:
1. **Lifted features**: x'_j = Σ_i A[i, j] · x_i (linear equality
   constraints). Captures both axis-aligned features (identity rows of
   A) and oblique combinations.
2. **Leaf disjunction**: exactly one leaf is active. We add one binary
   z_k per leaf and enforce Σ z_k = 1.
3. **Routing constraints**: for every internal node n with split on
   feature f_n at threshold t_n, the descendant-leaf indicators must
   respect the split:
     - If z_k corresponds to a leaf in n's left subtree:
         x'_{f_n} ≤ t_n  whenever  z_k = 1
     - similarly for right.
   We encode this with big-M:
     x'_{f_n} ≤ t_n + M · (1 − Σ_{k ∈ left(n)} z_k)
     x'_{f_n} ≥ t_n − M · (1 − Σ_{k ∈ right(n)} z_k)
4. **Leaf prediction**: y_t = Σ_k z_k · (β_k · x' + α_k). Encoded
   with auxiliary variables and big-M (since z is binary, the product
   is linearised exactly).

This corresponds to the standard "GDP via big-M" formulation also used
by OMLT, but built directly in discopt primitives (no Pyomo).

Returns a list of discopt Expression objects, one per output target.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from jax_ldt._types import Tree


def _check_discopt():
    try:
        import discopt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "discopt adapter requires `discopt`. Install with `pip install jax-ldt[discopt]`."
        ) from exc
    import discopt
    return discopt


def to_discopt_decision_tree(model_or_tree):
    """Convert a *constant-leaf* tree to `discopt.nn.tree.DecisionTree`.

    Raises if the tree has any non-trivial leaf regression coefficients
    (i.e., a true LMDT/HT). For those, use `embed_in_discopt_model`.
    """
    discopt = _check_discopt()
    from discopt.nn.tree import DecisionTree

    if isinstance(model_or_tree, Tree):
        tree = model_or_tree
    else:
        tree = getattr(model_or_tree, "tree_", None)
        if tree is None:
            raise ValueError("Pass a fitted model or a Tree pytree.")

    leaf_params = np.asarray(tree.leaf_params)  # (n_nodes, n_aug, T)
    is_leaf = np.asarray(tree.is_leaf)
    # For "constant-leaf" check: every leaf row must have only the bias entry nonzero.
    # leaf_params[node, 0, :] = bias; leaf_params[node, 1:, :] = weights.
    weights_nonzero = np.any(np.abs(leaf_params[:, 1:, :]) > 1e-12, axis=(1, 2))
    if np.any(weights_nonzero & is_leaf):
        raise ValueError(
            "Tree has non-trivial leaf regressions (LMDT / HT). "
            "discopt.nn.tree.DecisionTree only supports constant-leaf trees. "
            "Use `embed_in_discopt_model(model, m, x)` for a full hybrid embedding."
        )

    n_targets = int(tree.n_targets)
    if n_targets != 1:
        raise NotImplementedError(
            "discopt DecisionTree only supports single-target trees."
        )

    # Use bias-only value at each leaf
    feature = np.where(is_leaf, -1, np.asarray(tree.feature)).astype(np.int64)
    threshold = np.where(is_leaf, 0.0, np.asarray(tree.threshold)).astype(np.float64)
    left_child = np.where(is_leaf, -1, np.asarray(tree.left)).astype(np.int64)
    right_child = np.where(is_leaf, -1, np.asarray(tree.right)).astype(np.int64)
    value = leaf_params[:, 0, 0].astype(np.float64)

    # discopt's DecisionTree splits on n_features_transformed (the lifted space)
    return DecisionTree(
        n_features=int(tree.n_features_transformed),
        feature=feature,
        threshold=threshold,
        left_child=left_child,
        right_child=right_child,
        value=value,
    )


def _collect_leaves_in_subtree(tree: Tree, root_node: int) -> list[int]:
    is_leaf = np.asarray(tree.is_leaf)
    left = np.asarray(tree.left)
    right = np.asarray(tree.right)
    out: list[int] = []
    stack = [root_node]
    while stack:
        n = stack.pop()
        if is_leaf[n]:
            out.append(int(n))
        else:
            stack.append(int(left[n]))
            stack.append(int(right[n]))
    return out


def _build_descendant_leaves_map(tree: Tree) -> list[list[int]]:
    """Return ``descendants[nid]`` = list of leaf node ids reachable from ``nid``.

    Computed in a single post-order pass so a per-node descendant lookup
    is O(1) at constraint-emission time. Replaces an O(n_nodes·n_leaves)
    repeated DFS that previously fired once per internal node.
    """
    is_leaf = np.asarray(tree.is_leaf)
    left = np.asarray(tree.left)
    right = np.asarray(tree.right)
    n_nodes = is_leaf.shape[0]

    descendants: list[list[int]] = [[] for _ in range(n_nodes)]
    # Iterative post-order so we never recurse into Python's stack limit.
    visit_stack: list[tuple[int, bool]] = [(0, False)]
    while visit_stack:
        nid, processed = visit_stack.pop()
        if is_leaf[nid]:
            descendants[nid] = [int(nid)]
            continue
        if not processed:
            visit_stack.append((nid, True))
            visit_stack.append((int(right[nid]), False))
            visit_stack.append((int(left[nid]), False))
        else:
            descendants[nid] = (
                descendants[int(left[nid])] + descendants[int(right[nid])]
            )
    return descendants


def embed_in_discopt_model(
    model_or_tree,
    m,
    x_vars,
    *,
    big_m: Optional[float] = None,
    name_prefix: str = "tree",
):
    """Embed a trained tree into a `discopt.Model` as a hybrid disjunction.

    Parameters
    ----------
    model_or_tree : a fitted jax-ldt regressor or Tree pytree.
    m             : `discopt.Model` to add variables/constraints to.
    x_vars        : a discopt Variable / Expression of shape
                    (n_features_in,). Use `m.continuous("x", shape=(n,))`
                    to create one.
    big_m         : optional override for the big-M constant. If None,
                    we compute one from feature ranges in the lifted
                    space (max - min of A · bound_box) + 1e-3.
    name_prefix   : prefix for auto-generated variables.

    Returns
    -------
    y_expr : list of `discopt.Expression`, one per output target.
             For the common single-target case, you can use `y_expr[0]`.
    """
    discopt = _check_discopt()

    if isinstance(model_or_tree, Tree):
        tree = model_or_tree
    else:
        tree = getattr(model_or_tree, "tree_", None)
        if tree is None:
            raise ValueError("Pass a fitted model or a Tree pytree.")

    n_in = int(tree.n_features_in)
    n_t = int(tree.n_features_transformed)
    n_targets = int(tree.n_targets)
    A = np.asarray(tree.transform_matrix)

    # Lifted features: x'_j = Σ_i A[i,j] x_i
    x_t = m.continuous(f"{name_prefix}_xt", shape=(n_t,))
    for j in range(n_t):
        # x_t[j] - sum_i A[i,j] x_vars[i] == 0
        expr = x_t[j]
        for i in range(n_in):
            coeff = float(A[i, j])
            if coeff != 0.0:
                expr = expr - coeff * x_vars[i]
        m.subject_to(expr == 0.0)

    # Leaves and their parameters
    is_leaf = np.asarray(tree.is_leaf)
    leaf_ids = np.where(is_leaf)[0]
    n_leaves = len(leaf_ids)
    leaf_params = np.asarray(tree.leaf_params)  # (n_nodes, n_aug, T)

    # Binary leaf indicators
    z = m.binary(f"{name_prefix}_z", shape=(n_leaves,))
    # Exactly one active
    m.subject_to(sum(z[i] for i in range(n_leaves)) == 1)

    # Bounds-driven big-M:
    #   lb_t[j], ub_t[j] = bounds on x'_j = Σ_i A[i,j] x_i induced by the
    #   bounds on x. Used for:
    #     - per-node routing M, which must dominate |x'_f − t|
    #     - per-leaf output M, which must dominate |α_k + β_k·x'|
    #   When x_vars exposes no usable bounds and the user did not override
    #   big_m, we fall back to a single large constant and warn.
    fallback = 1e6 if big_m is None else float(big_m)
    margin = 1e-3
    have_bounds = False
    if big_m is None:
        try:
            lb = np.asarray(getattr(x_vars, "lb", None), dtype=np.float64)
            ub = np.asarray(getattr(x_vars, "ub", None), dtype=np.float64)
            if (
                lb.size == n_in
                and ub.size == n_in
                and np.all(np.isfinite(lb))
                and np.all(np.isfinite(ub))
            ):
                have_bounds = True
            else:
                warnings.warn(
                    f"embed_in_discopt_model: x_vars bounds not finite or wrong size "
                    f"(lb.size={lb.size}, ub.size={ub.size}, expected {n_in}); "
                    f"falling back to big_m={fallback}. Pass big_m=... explicitly "
                    f"for a tighter formulation.",
                    UserWarning,
                    stacklevel=2,
                )
        except (AttributeError, TypeError):
            warnings.warn(
                f"embed_in_discopt_model: x_vars does not expose lb/ub; "
                f"falling back to big_m={fallback}. Pass big_m=... explicitly "
                f"for a tighter formulation.",
                UserWarning,
                stacklevel=2,
            )

    if have_bounds:
        # x'_j = Σ_i A[i,j] x_i; with x ∈ [lb, ub] this gives
        #   lb_t[j] = Σ_i (A[i,j] · (lb if A>=0 else ub))
        #   ub_t[j] = Σ_i (A[i,j] · (ub if A>=0 else lb))
        At = A.T  # (n_t, n_in)
        At_pos = np.maximum(At, 0.0)
        At_neg = np.minimum(At, 0.0)
        lb_t = At_pos @ lb + At_neg @ ub
        ub_t = At_pos @ ub + At_neg @ lb
    else:
        lb_t = np.full(n_t, -fallback, dtype=np.float64)
        ub_t = np.full(n_t, fallback, dtype=np.float64)

    # Routing constraints. For every internal node, every leaf indicator
    # must enforce the split on its lifted feature.
    feature_arr = np.asarray(tree.feature)
    threshold_arr = np.asarray(tree.threshold)
    left_arr = np.asarray(tree.left)
    right_arr = np.asarray(tree.right)
    leaf_id_to_z = {int(lid): i for i, lid in enumerate(leaf_ids)}

    n_nodes = is_leaf.shape[0]
    descendants = _build_descendant_leaves_map(tree)
    for nid in range(n_nodes):
        if is_leaf[nid]:
            continue
        f = int(feature_arr[nid])
        t = float(threshold_arr[nid])
        left_leaves = descendants[int(left_arr[nid])]
        right_leaves = descendants[int(right_arr[nid])]

        # Per-side big-M: worst-case |x'_f − t| under bounds.
        m_left = float(max(ub_t[f] - t, 0.0)) + margin
        m_right = float(max(t - lb_t[f], 0.0)) + margin

        sum_left = sum(z[leaf_id_to_z[lid]] for lid in left_leaves)
        sum_right = sum(z[leaf_id_to_z[lid]] for lid in right_leaves)
        m.subject_to(x_t[f] - t <= m_left * (1 - sum_left))
        m.subject_to(t - x_t[f] <= m_right * (1 - sum_right))

    # Output expression: y = Σ_k z_k · (β_k · x' + α_k)
    # β_k · x' + α_k where leaf_params[k, :, t] = [α_k, β_k_1, β_k_2, ...].
    # Since z_k is binary and (β · x' + α) is bounded linear, the product
    # z_k · linear is linearisable via auxiliary continuous vars and big-M.
    # The bound on |β · x' + α| over x' ∈ [lb_t, ub_t] is computed
    # per-leaf so a tight M_k is used for the v_k ≤ M_k z_k constraints.
    y_exprs = []
    for tgt in range(n_targets):
        v = m.continuous(f"{name_prefix}_v_t{tgt}", shape=(n_leaves,))
        for k, lid in enumerate(leaf_ids):
            params = leaf_params[lid, :, tgt]
            alpha = float(params[0])
            betas = params[1:].astype(np.float64)

            lin_expr = alpha
            for j in range(n_t):
                if betas[j] != 0.0:
                    lin_expr = lin_expr + float(betas[j]) * x_t[j]

            # Per-leaf output bounds: |α + Σ_j β_j x'_j| ≤ M_k.
            beta_pos = np.maximum(betas, 0.0)
            beta_neg = np.minimum(betas, 0.0)
            lin_lb = alpha + float(beta_pos @ lb_t + beta_neg @ ub_t)
            lin_ub = alpha + float(beta_pos @ ub_t + beta_neg @ lb_t)
            m_k = max(abs(lin_lb), abs(lin_ub)) + margin

            # Big-M constraints linking v_k to z_k * lin_expr.
            m.subject_to(v[k] <= m_k * z[k])
            m.subject_to(v[k] >= -m_k * z[k])
            m.subject_to(v[k] - lin_expr <= m_k * (1 - z[k]))
            m.subject_to(lin_expr - v[k] <= m_k * (1 - z[k]))

        y_t = sum(v[k] for k in range(n_leaves))
        y_exprs.append(y_t)

    return y_exprs
