"""Adapter from jax-ldt Tree → Pyomo model embedding.

`embed_in_pyomo_model(model, m, x)` adds a sub-block to a Pyomo
`ConcreteModel` (or any `Block`) containing the variables and
constraints that express the trained tree's output exactly as a MILP,
and returns one Pyomo variable per output target that can be used in
objectives or other constraints.

The formulation is the same "GDP via big-M" encoding used by
`embed_in_discopt_model` (see `discopt_adapter.py` for the derivation):

1. lifted features  x'_j = Σ_i A[i, j] · x_i
2. one binary z_k per leaf with Σ z_k = 1
3. big-M routing on every internal split
     x'_f − t       ≤ M · (1 − Σ_{k ∈ left(n)}  z_k)
     t + eps − x'_f ≤ M · (1 − Σ_{k ∈ right(n)} z_k)
4. y = Σ_k v_k with v_k = z_k · (α_k + β_k · x') linearised by big-M.

Everything lives on `m.<name>` so several trees can be embedded in the
same model without name clashes. Pyomo is an optional dependency.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from jax_ldt._types import Tree
from jax_ldt.export.discopt_adapter import _build_descendant_leaves_map


def _check_pyomo():
    try:
        import pyomo.environ as pyo
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyomo adapter requires `pyomo`. Install with `pip install jax-ldt[pyomo]`."
        ) from exc
    return pyo


def _as_var_list(x_vars) -> list:
    """Flatten an IndexedVar (in index order) or a sequence into a list."""
    if hasattr(x_vars, "values") and callable(x_vars.values):
        return list(x_vars.values())
    return list(x_vars)


def _bounds_of(xs: list, n_in: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Finite (lb, ub) arrays for `xs`, or None if any bound is missing.

    Pyomo reports an unbounded side as ``None``; expressions (rather than
    variables) have no ``lb``/``ub`` at all. Anything with |bound| > 1e10
    is treated as unbounded since it is numerically toxic as a big-M.
    """
    if len(xs) != n_in:
        raise ValueError(f"x_vars has {len(xs)} entries; the tree expects {n_in}.")
    lb = [getattr(v, "lb", None) for v in xs]
    ub = [getattr(v, "ub", None) for v in xs]
    if any(b is None for b in lb + ub):
        return None
    lb_a = np.asarray(lb, dtype=np.float64)
    ub_a = np.asarray(ub, dtype=np.float64)
    if max(np.max(np.abs(lb_a)), np.max(np.abs(ub_a))) > 1e10:
        return None
    return lb_a, ub_a


def embed_in_pyomo_model(
    model_or_tree,
    m,
    x_vars,
    *,
    big_m: Optional[float] = None,
    boundary_eps: float = 0.0,
    name: str = "tree",
):
    """Embed a trained tree into a Pyomo model as an exact big-M MILP.

    Parameters
    ----------
    model_or_tree : a fitted jax-ldt regressor or Tree pytree.
    m             : Pyomo `ConcreteModel` / `Block` to add the sub-block to.
    x_vars        : the n_features_in inputs, as a Pyomo IndexedVar (taken
                    in index order) or a sequence of Pyomo variables /
                    linear expressions.
    big_m         : optional big-M *constant*. If None (recommended),
                    per-node and per-leaf big-Ms are derived from the
                    bounds on `x_vars` (tightest valid formulation). If
                    given, it is used for every routing and leaf-output
                    constraint; when `x_vars` are bounded, a `big_m` too
                    small to be valid raises `ValueError`.
    boundary_eps  : strict-inequality margin for the right branch, default
                    0 (off). See `embed_in_discopt_model` for the
                    tie-breaking trade-off.
    name          : name of the sub-block added to `m`.

    Returns
    -------
    y : list of Pyomo variables, one per output target (`m.<name>.y[t]`).
        For the common single-target case use `y[0]`.
    """
    from jax_ldt._validation import _require_x64

    _require_x64()
    pyo = _check_pyomo()

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
    xs = _as_var_list(x_vars)

    # Lifted-space bounds: x'_j = Σ_i A[i,j] x_i with x ∈ [lb, ub].
    user_big_m = None if big_m is None else float(big_m)
    margin = 1e-3
    bounds = _bounds_of(xs, n_in)
    have_bounds = bounds is not None
    if bounds is not None:
        lb, ub = bounds
        At_pos = np.maximum(A.T, 0.0)
        At_neg = np.minimum(A.T, 0.0)
        lb_t = At_pos @ lb + At_neg @ ub
        ub_t = At_pos @ ub + At_neg @ lb
    elif user_big_m is not None:
        warnings.warn(
            f"embed_in_pyomo_model: big_m={user_big_m} cannot be validated "
            "because x_vars do not all have finite bounds; if it is smaller "
            "than some split threshold or leaf-output magnitude the MIP will "
            "silently cut off feasible leaves. Bound x_vars so big_m can be "
            "checked, or ensure big_m comfortably exceeds the feature scale.",
            UserWarning,
            stacklevel=2,
        )
    else:
        fallback = 1e6
        warnings.warn(
            "embed_in_pyomo_model: x_vars do not all have finite bounds; "
            f"falling back to big_m={fallback}. Pass big_m=... explicitly, or "
            "bound x_vars, for a tighter and safer formulation.",
            UserWarning,
            stacklevel=2,
        )
        lb_t = np.full(n_t, -fallback)
        ub_t = np.full(n_t, fallback)
        have_bounds = True  # derive per-node Ms from the fallback box

    is_leaf = np.asarray(tree.is_leaf)
    leaf_ids = [int(i) for i in np.where(is_leaf)[0]]
    n_leaves = len(leaf_ids)
    leaf_params = np.asarray(tree.leaf_params)  # (n_nodes, n_aug, T)
    leaf_id_to_z = {lid: k for k, lid in enumerate(leaf_ids)}

    b = pyo.Block(concrete=True)
    m.add_component(name, b)

    b.xt = pyo.Var(range(n_t), domain=pyo.Reals)
    if bounds is not None:
        for j in range(n_t):
            b.xt[j].setlb(float(lb_t[j]))
            b.xt[j].setub(float(ub_t[j]))
    b.lift = pyo.Constraint(
        range(n_t),
        rule=lambda _b, j: _b.xt[j]
        == sum(float(A[i, j]) * xs[i] for i in range(n_in) if A[i, j] != 0.0),
    )

    b.z = pyo.Var(range(n_leaves), domain=pyo.Binary)
    b.one_leaf = pyo.Constraint(expr=sum(b.z[k] for k in range(n_leaves)) == 1)

    # Routing constraints, one pair per internal node.
    feature_arr = np.asarray(tree.feature)
    threshold_arr = np.asarray(tree.threshold)
    left_arr = np.asarray(tree.left)
    right_arr = np.asarray(tree.right)
    descendants = _build_descendant_leaves_map(tree)
    internal = [int(n) for n in np.where(~is_leaf)[0]]
    b.route_left = pyo.Constraint(internal)
    b.route_right = pyo.Constraint(internal)
    for nid in internal:
        f = int(feature_arr[nid])
        t = float(threshold_arr[nid])
        if user_big_m is not None:
            if bounds is not None:
                req = max(float(ub_t[f]) - t, t + boundary_eps - float(lb_t[f]), 0.0)
                if user_big_m < req:
                    raise ValueError(
                        f"big_m={user_big_m} is too small: node {nid} splitting "
                        f"lifted feature {f} at t={t:g} needs a routing big-M of "
                        f"at least {req:g} to avoid cutting off feasible leaves. "
                        "Increase big_m or pass tighter x_vars bounds."
                    )
            m_left = m_right = user_big_m
        else:
            m_left = float(max(ub_t[f] - t, 0.0)) + margin
            m_right = float(max(t + boundary_eps - lb_t[f], 0.0)) + margin

        sum_left = sum(b.z[leaf_id_to_z[k]] for k in descendants[int(left_arr[nid])])
        sum_right = sum(b.z[leaf_id_to_z[k]] for k in descendants[int(right_arr[nid])])
        # In-memory routing sends x'_f == t LEFT.
        b.route_left[nid] = b.xt[f] - t <= m_left * (1 - sum_left)
        b.route_right[nid] = t + boundary_eps - b.xt[f] <= m_right * (1 - sum_right)

    # Leaf outputs: v[t, k] = z_k · (α_k + β_k · x'), linearised with big-M.
    b.v = pyo.Var(range(n_targets), range(n_leaves), domain=pyo.Reals)
    b.y = pyo.Var(range(n_targets), domain=pyo.Reals)
    b.leaf_out = pyo.ConstraintList()
    for tgt in range(n_targets):
        for k, lid in enumerate(leaf_ids):
            params = leaf_params[lid, :, tgt]
            alpha = float(params[0])
            betas = params[1:].astype(np.float64)
            lin_expr = alpha + sum(
                float(betas[j]) * b.xt[j] for j in range(n_t) if betas[j] != 0.0
            )

            if have_bounds:
                beta_pos = np.maximum(betas, 0.0)
                beta_neg = np.minimum(betas, 0.0)
                lin_lb = alpha + float(beta_pos @ lb_t + beta_neg @ ub_t)
                lin_ub = alpha + float(beta_pos @ ub_t + beta_neg @ lb_t)
                req = max(abs(lin_lb), abs(lin_ub))
            if user_big_m is not None:
                if bounds is not None and user_big_m < req:
                    raise ValueError(
                        f"big_m={user_big_m} is too small: leaf {lid} "
                        f"(target {tgt}) has output magnitude up to {req:g}; "
                        "the v_k linearization needs a big-M at least that "
                        "large. Increase big_m or pass tighter x_vars bounds."
                    )
                m_k = user_big_m
            else:
                m_k = req + margin

            v = b.v[tgt, k]
            b.leaf_out.add(v <= m_k * b.z[k])
            b.leaf_out.add(v >= -m_k * b.z[k])
            b.leaf_out.add(v - lin_expr <= m_k * (1 - b.z[k]))
            b.leaf_out.add(lin_expr - v <= m_k * (1 - b.z[k]))

    b.output = pyo.Constraint(
        range(n_targets),
        rule=lambda _b, tgt: _b.y[tgt] == sum(_b.v[tgt, k] for k in range(n_leaves)),
    )
    return [b.y[tgt] for tgt in range(n_targets)]
