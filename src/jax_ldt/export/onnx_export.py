"""Hand-built ONNX export for trained trees.

Strategy: encode the tree as a custom subgraph that:
1. Lifts X via MatMul with the transform matrix → X_t (shape: N × n_transformed)
2. Augments by prepending a bias column of ones → X_aug (N × n_aug)
3. Routes each row through the tree to find its leaf id (using a sequence
   of comparison + Where ops over node arrays)
4. Selects the leaf's params and computes X_aug · params → ŷ

We do NOT use ONNX-ML's TreeEnsembleRegressor: that op encodes a tree
with constant leaf values, while our leaves are linear models in the
lifted space. So we synthesize the routing in core ONNX ops (Less,
Where, Gather), giving a graph that any standard runtime can execute.

For deep trees this implementation unrolls the routing as a loop over
depth, which keeps the graph fully feed-forward and easy to inspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from jax_ldt._types import Tree


def _import_onnx():
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ONNX export requires `onnx`. Install with `pip install jax-ldt[onnx]`."
        ) from exc
    return onnx, TensorProto, helper, numpy_helper


def to_onnx(
    model_or_tree, path: Optional[Union[str, Path]] = None, *, opset: int = 18
):
    """Build an ONNX ModelProto for the trained tree.

    Parameters
    ----------
    model_or_tree : a fitted model with `.tree_`, or a Tree pytree.
    path : optional output path; if given, also writes the model.
    opset : ONNX opset version. We need >= 13 for ScatterND-style indexing.

    Returns
    -------
    onnx.ModelProto
    """
    onnx, TensorProto, helper, numpy_helper = _import_onnx()

    if isinstance(model_or_tree, Tree):
        tree = model_or_tree
    else:
        tree = getattr(model_or_tree, "tree_", None)
        if tree is None:
            raise ValueError("Pass a fitted model (with `.tree_`) or a Tree pytree.")

    n_in = int(tree.n_features_in)
    n_t = int(tree.n_features_transformed)
    n_aug = n_t + 1
    n_targets = int(tree.n_targets)
    n_nodes = int(tree.is_leaf.shape[0])

    # Compute max routing depth so we can fix the ONNX graph length.
    max_depth = _compute_max_depth(tree)

    A = np.asarray(tree.transform_matrix, dtype=np.float64)  # (n_in, n_t)
    feature = np.asarray(tree.feature, dtype=np.int64)  # (n_nodes,)
    threshold = np.asarray(tree.threshold, dtype=np.float64)
    threshold = np.where(np.isnan(threshold), 0.0, threshold)
    left = np.asarray(tree.left, dtype=np.int64)
    right = np.asarray(tree.right, dtype=np.int64)
    leaf_params = np.asarray(tree.leaf_params, dtype=np.float64)  # (n_nodes, n_aug, T)

    # For leaves, set left = right = own id so routing fixed-points there.
    is_leaf_np = np.asarray(tree.is_leaf, dtype=bool)
    own_ids = np.arange(n_nodes, dtype=np.int64)
    left_safe = np.where(is_leaf_np, own_ids, left)
    right_safe = np.where(is_leaf_np, own_ids, right)
    feature_safe = np.where(feature < 0, 0, feature)  # safe gather index

    # Initializers (constants baked into the model)
    inits = []
    inits.append(numpy_helper.from_array(A.astype(np.float64), name="transform_matrix"))
    inits.append(numpy_helper.from_array(feature_safe, name="feature"))
    inits.append(numpy_helper.from_array(threshold.astype(np.float64), name="threshold"))
    inits.append(numpy_helper.from_array(left_safe, name="left"))
    inits.append(numpy_helper.from_array(right_safe, name="right"))
    inits.append(numpy_helper.from_array(leaf_params.astype(np.float64), name="leaf_params"))
    inits.append(numpy_helper.from_array(np.array([0.0], dtype=np.float64), name="zero_scalar"))
    inits.append(numpy_helper.from_array(np.array([1.0], dtype=np.float64), name="one_scalar"))
    # `axes_1` and `axes_0` are reused as the `axes` input to Squeeze /
    # Unsqueeze (opset ≥ 13) — i.e. the axis *index*, not a value of 1
    # being broadcast. The "ones_int_1d" alias is preserved for
    # backwards compatibility with cached graphs but the meaning is
    # "axes = [1]".
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name="ones_int_1d"))
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_int_1d"))

    nodes = []

    # Input: X with shape (N, n_in)
    # Step 1: lift to transformed space
    nodes.append(helper.make_node("MatMul", inputs=["X", "transform_matrix"], outputs=["X_t"]))

    # Step 2: augment with bias column.
    # ones_col = (N, 1) of ones — built via Shape/Gather/Expand for dynamic N.
    nodes.append(helper.make_node("Shape", inputs=["X"], outputs=["X_shape"]))
    # Pull X.shape[0] (the batch dimension) by gathering at index 0.
    nodes.append(
        helper.make_node(
            "Gather",
            inputs=["X_shape", "zero_int_1d"],
            outputs=["N_dim"],
            axis=0,
        )
    )
    # Build target shape = (N, 1)
    nodes.append(
        helper.make_node(
            "Concat", inputs=["N_dim", "ones_int_1d"], outputs=["N_one_shape"], axis=0
        )
    )
    nodes.append(
        helper.make_node(
            "Expand", inputs=["one_scalar", "N_one_shape"], outputs=["ones_col"]
        )
    )
    nodes.append(
        helper.make_node(
            "Concat", inputs=["ones_col", "X_t"], outputs=["X_aug"], axis=1
        )
    )

    # Step 3: route. Start with leaf_id = zeros(N) (root = id 0).
    nodes.append(
        helper.make_node(
            "Expand",
            inputs=["zero_scalar", "N_dim"],
            outputs=["leaf_id_f"],
        )
    )
    nodes.append(
        helper.make_node("Cast", inputs=["leaf_id_f"], outputs=["leaf_id_0"], to=TensorProto.INT64)
    )

    # Iterate: for each depth step, gather feature/threshold/left/right
    # at current leaf_id, compare X_t[i, feat[id]] <= thr[id], then update.
    cur = "leaf_id_0"
    for step in range(max_depth):
        feat_name = f"feat_{step}"
        thr_name = f"thr_{step}"
        left_name = f"left_{step}"
        right_name = f"right_{step}"
        feat_val_name = f"feat_val_{step}"
        cmp_name = f"cmp_{step}"
        nxt_name = f"leaf_id_{step + 1}"

        nodes.append(helper.make_node("Gather", inputs=["feature", cur], outputs=[feat_name], axis=0))
        nodes.append(helper.make_node("Gather", inputs=["threshold", cur], outputs=[thr_name], axis=0))
        nodes.append(helper.make_node("Gather", inputs=["left", cur], outputs=[left_name], axis=0))
        nodes.append(helper.make_node("Gather", inputs=["right", cur], outputs=[right_name], axis=0))

        # X_t has shape (N, n_t). For each row i, we need X_t[i, feat[i]].
        # Use GatherElements on axis 1 with index = feat (shape (N,)).
        # GatherElements requires shape (N, 1), so unsqueeze.
        nodes.append(
            helper.make_node("Unsqueeze", inputs=[feat_name, "ones_int_1d"], outputs=[f"{feat_name}_u"])
        )
        nodes.append(
            helper.make_node(
                "GatherElements",
                inputs=["X_t", f"{feat_name}_u"],
                outputs=[f"{feat_val_name}_2d"],
                axis=1,
            )
        )
        nodes.append(
            helper.make_node(
                "Squeeze", inputs=[f"{feat_val_name}_2d", "ones_int_1d"], outputs=[feat_val_name]
            )
        )

        # `LessOrEqual` (left-on-equality) matches `tree_core._route_one`
        # and the splitter in `_evaluate_splits_kernel`. Changing this to
        # `Less` would produce off-by-one routing on points that fall
        # exactly on a split threshold.
        nodes.append(
            helper.make_node("LessOrEqual", inputs=[feat_val_name, thr_name], outputs=[cmp_name])
        )
        nodes.append(
            helper.make_node(
                "Where", inputs=[cmp_name, left_name, right_name], outputs=[nxt_name]
            )
        )
        cur = nxt_name

    # Step 4: gather leaf params per row, then einsum-equivalent
    # leaf_params: (n_nodes, n_aug, T); per-row params: (N, n_aug, T)
    nodes.append(helper.make_node("Gather", inputs=["leaf_params", cur], outputs=["params_per"], axis=0))

    # einsum "nf,nft->nt": broadcast X_aug (N, n_aug) → (N, n_aug, 1), multiply, reduce
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name="two_int_1d"))
    nodes.append(
        helper.make_node("Unsqueeze", inputs=["X_aug", "two_int_1d"], outputs=["X_aug_3d"])
    )
    # X_aug_3d: (N, n_aug, 1); params_per: (N, n_aug, T) → product (N, n_aug, T) → reduce axis=1
    nodes.append(helper.make_node("Mul", inputs=["X_aug_3d", "params_per"], outputs=["prod"]))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name="reduce_axes_1"))
    nodes.append(
        helper.make_node(
            "ReduceSum", inputs=["prod", "reduce_axes_1"], outputs=["yh_3d"], keepdims=0
        )
    )

    output_name = "Y"
    if n_targets == 1:
        # squeeze trailing axis to give (N,)
        nodes.append(
            helper.make_node("Squeeze", inputs=["yh_3d", "ones_int_1d"], outputs=[output_name])
        )
    else:
        nodes.append(helper.make_node("Identity", inputs=["yh_3d"], outputs=[output_name]))

    # IO definitions
    X_value_info = helper.make_tensor_value_info("X", TensorProto.DOUBLE, ["N", n_in])
    if n_targets == 1:
        Y_value_info = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, ["N"])
    else:
        Y_value_info = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, ["N", n_targets])

    graph = helper.make_graph(
        nodes,
        name="jax_ldt_tree",
        inputs=[X_value_info],
        outputs=[Y_value_info],
        initializer=inits,
    )

    opset_imports = [helper.make_opsetid("", opset)]
    # Let `helper.make_model` derive a compatible ir_version from the
    # opset rather than hardcoding one. Older code pinned ``ir_version=8``,
    # which produced checker warnings on newer onnx releases (where opset
    # 18 expects ir_version ≥ 9). With no override, the helper picks the
    # minimum ir_version the requested opset is valid against.
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.producer_name = "jax-ldt"
    onnx.checker.check_model(model)

    if path is not None:
        onnx.save(model, str(path))

    return model


def _compute_max_depth(tree: Tree) -> int:
    """Maximum number of decisions to reach any leaf (root is depth 0)."""
    is_leaf = np.asarray(tree.is_leaf)
    left = np.asarray(tree.left)
    right = np.asarray(tree.right)
    n = is_leaf.shape[0]
    depth = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    stack = [(0, 0)]
    max_d = 0
    while stack:
        nid, d = stack.pop()
        if visited[nid]:
            continue
        visited[nid] = True
        depth[nid] = d
        max_d = max(max_d, d)
        if not is_leaf[nid]:
            l_id = int(left[nid])
            r_id = int(right[nid])
            if l_id >= 0:
                stack.append((l_id, d + 1))
            if r_id >= 0:
                stack.append((r_id, d + 1))
    return int(max_d)
