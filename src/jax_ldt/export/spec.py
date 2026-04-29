"""Solver-neutral JSON serialization for trained trees.

Schema (v1):
{
  "version": 1,
  "kind": "linear_tree" | "hyperplane_tree",
  "n_features_in": int,
  "n_features_transformed": int,
  "n_targets": int,
  "transform_matrix": [[...]],
  "categorical_features": [int, ...],
  "linear_features": [int, ...],
  "tree": {
    "node_count": int,
    "is_leaf":   [bool, ...],
    "feature":   [int, ...],
    "threshold": [float, ...],
    "left":      [int, ...],
    "right":     [int, ...],
    "leaf_params": [[[float]]]   # (n_nodes, n_aug, n_targets)
  },
  "uq": {                          # optional
    "method": "linprop",
    "leaf_n":     [int, ...],
    "leaf_x_mean":[[...]],
    "leaf_x_var": [[...]],
    "leaf_mse":   [[...]]
  }
}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import jax.numpy as jnp
import numpy as np

from jax_ldt._types import LeafUQ, Tree


SPEC_VERSION = 1


def _np(x: Any) -> Any:
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _none_for_nan(arr: np.ndarray) -> Any:
    """Recursively convert an n-D float array to a nested list, with
    every NaN replaced by ``None``.

    Plain ``arr.tolist()`` produces Python ``float('nan')`` values which
    Python's ``json`` module renders as the literal ``NaN`` — invalid by
    the JSON spec and rejected by strict parsers (e.g., ``json.loads`` in
    other languages, ``allow_nan=False`` round-trips). Encoding NaN as
    ``null`` keeps the payload spec-compliant.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return None if (arr.dtype.kind == "f" and np.isnan(arr)) else arr.item()
    return [_none_for_nan(sub) for sub in arr]


def tree_to_dict(tree: Tree, *, kind: str = "tree") -> dict[str, Any]:
    """Serialize a Tree pytree to a plain Python dict ready for JSON dump.

    Parameters
    ----------
    tree : Tree
        Fitted tree pytree (from a regressor's ``.tree_`` attribute).
    kind : str, keyword-only, default ``"tree"``
        Tag written to the ``"kind"`` field of the JSON payload.
        Standard values are ``"linear_tree"`` (axis-aligned LMDT) and
        ``"hyperplane_tree"`` (oblique-split HT). Downstream consumers
        may branch on this string to dispatch the right loader. When
        :func:`to_json` is called with a fitted regressor, ``kind`` is
        inferred automatically; pass it explicitly here when you have
        only the bare ``Tree`` pytree.
    """
    threshold = np.asarray(tree.threshold)
    # JSON has no NaN; encode as null on the way out.
    threshold_list: list[Optional[float]] = [
        None if math.isnan(v) else float(v) for v in threshold.tolist()
    ]

    out: dict[str, Any] = {
        "version": SPEC_VERSION,
        "kind": kind,
        "n_features_in": int(tree.n_features_in),
        "n_features_transformed": int(tree.n_features_transformed),
        "n_targets": int(tree.n_targets),
        "transform_matrix": _np(np.asarray(tree.transform_matrix)),
        "categorical_features": list(tree.categorical_features),
        "linear_features": list(tree.linear_features),
        "tree": {
            "node_count": int(tree.is_leaf.shape[0]),
            "is_leaf": [bool(v) for v in np.asarray(tree.is_leaf).tolist()],
            "feature": [int(v) for v in np.asarray(tree.feature).tolist()],
            "threshold": threshold_list,
            "left": [int(v) for v in np.asarray(tree.left).tolist()],
            "right": [int(v) for v in np.asarray(tree.right).tolist()],
            "leaf_params": _none_for_nan(np.asarray(tree.leaf_params)),
        },
    }

    if tree.leaf_uq is not None:
        out["uq"] = {
            "method": "linprop",
            "leaf_n": [int(v) for v in np.asarray(tree.leaf_uq.n).tolist()],
            "leaf_x_mean": _np(np.asarray(tree.leaf_uq.x_mean)),
            "leaf_x_var": _np(np.asarray(tree.leaf_uq.x_var)),
            "leaf_mse": _np(np.asarray(tree.leaf_uq.mse)),
        }
    return out


def _nan_for_none_array(nested: Any) -> np.ndarray:
    """Inverse of :func:`_none_for_nan`: build a float array where ``None``
    entries become ``NaN``. Accepts arbitrary nesting depth."""
    arr = np.array(nested, dtype=object)
    out = np.empty(arr.shape, dtype=np.float64)
    for idx in np.ndindex(arr.shape):
        v = arr[idx]
        out[idx] = float("nan") if v is None else float(v)
    return out


def tree_from_dict(d: dict[str, Any]) -> Tree:
    """Rebuild a Tree from a JSON-compatible dict."""
    if d.get("version") != SPEC_VERSION:
        raise ValueError(f"Unsupported spec version: {d.get('version')}")
    t = d["tree"]
    threshold_list = t["threshold"]
    threshold = np.asarray(
        [float("nan") if v is None else float(v) for v in threshold_list], dtype=np.float64
    )
    leaf_params = _nan_for_none_array(t["leaf_params"])
    leaf_uq = None
    if "uq" in d:
        u = d["uq"]
        leaf_uq = LeafUQ(
            n=jnp.asarray(u["leaf_n"], dtype=jnp.int32),
            x_mean=jnp.asarray(u["leaf_x_mean"], dtype=jnp.float64),
            x_var=jnp.asarray(u["leaf_x_var"], dtype=jnp.float64),
            mse=jnp.asarray(u["leaf_mse"], dtype=jnp.float64),
        )

    return Tree(
        is_leaf=jnp.asarray(t["is_leaf"], dtype=jnp.bool_),
        feature=jnp.asarray(t["feature"], dtype=jnp.int32),
        threshold=jnp.asarray(threshold),
        left=jnp.asarray(t["left"], dtype=jnp.int32),
        right=jnp.asarray(t["right"], dtype=jnp.int32),
        leaf_params=jnp.asarray(leaf_params, dtype=jnp.float64),
        transform_matrix=jnp.asarray(d["transform_matrix"], dtype=jnp.float64),
        leaf_uq=leaf_uq,
        n_features_in=int(d["n_features_in"]),
        n_features_transformed=int(d["n_features_transformed"]),
        n_targets=int(d["n_targets"]),
        linear_features=tuple(int(v) for v in d["linear_features"]),
        categorical_features=tuple(int(v) for v in d["categorical_features"]),
    )


_KIND_BY_CLASSNAME = {
    "LinearTreeRegressor": "linear_tree",
    "HyperplaneTreeRegressor": "hyperplane_tree",
}


def _infer_kind(model_or_tree) -> str:
    """Infer a kind string from the regressor class.

    For raw `Tree` pytrees we cannot tell axis-aligned from oblique by
    inspection (LMDT trees have an identity transform_matrix; HT trees
    have a non-identity one), so we fall back to that heuristic.
    """
    if isinstance(model_or_tree, Tree):
        # Identity transform → axis-aligned LMDT; otherwise hyperplane.
        n_in = int(model_or_tree.n_features_in)
        n_t = int(model_or_tree.n_features_transformed)
        if n_in == n_t:
            tm = model_or_tree.transform_matrix
            import jax.numpy as _jnp
            if bool(_jnp.allclose(tm, _jnp.eye(n_in, dtype=tm.dtype))):
                return "linear_tree"
        return "hyperplane_tree"
    return _KIND_BY_CLASSNAME.get(type(model_or_tree).__name__, "tree")


def to_json(
    model_or_tree,
    path: Optional[Union[str, Path]] = None,
    *,
    kind: Optional[str] = None,
) -> str:
    """Serialize a fitted model or a Tree to JSON.

    Parameters
    ----------
    model_or_tree : a fitted regressor (with ``.tree_``) or a `Tree` pytree.
    path : optional output path; if given, the JSON is also written to disk.
    kind : optional explicit ``"linear_tree"`` / ``"hyperplane_tree"`` /
        any other label. If ``None`` (default), the kind is inferred from
        the regressor class (or, for raw `Tree` inputs, from whether the
        transform matrix is the identity).

    Always returns the JSON string.
    """
    if isinstance(model_or_tree, Tree):
        tree = model_or_tree
    else:
        tree = getattr(model_or_tree, "tree_", None)
        if tree is None:
            raise ValueError("Pass a fitted model (with `.tree_`) or a Tree pytree.")
    if kind is None:
        kind = _infer_kind(model_or_tree)
    payload = tree_to_dict(tree, kind=kind)
    s = json.dumps(payload, indent=2)
    if path is not None:
        Path(path).write_text(s)
    return s


def from_json(source: Union[str, Path]) -> Tree:
    """Deserialize a Tree from a JSON string or file path.

    A ``Path`` is always read from disk. A ``str`` is treated as raw JSON
    when it begins with ``{`` (after whitespace); otherwise it is treated
    as a path. Strings that look like a path but do not exist raise
    :class:`FileNotFoundError` rather than falling through to the JSON
    parser, which would otherwise emit a confusing ``JSONDecodeError``.
    """
    if isinstance(source, Path):
        s = source.read_text()
    elif isinstance(source, str):
        if source.lstrip().startswith("{"):
            s = source
        else:
            # Treat as a path; a non-existent path is a clear error here
            # (rather than silently falling through to JSON parsing,
            # which would surface a misleading "Expecting value" message).
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(
                    f"from_json: source does not look like JSON text and the "
                    f"path {source!r} does not exist."
                )
            s = p.read_text()
    else:
        raise TypeError(
            f"from_json: source must be a str or Path; got {type(source).__name__}"
        )
    payload = json.loads(s)
    return tree_from_dict(payload)
