"""Export utilities: solver-neutral JSON + ONNX."""

from __future__ import annotations

from jax_ldt.export.spec import from_json, to_json, tree_from_dict, tree_to_dict

__all__ = ["to_json", "from_json", "tree_to_dict", "tree_from_dict"]


def to_onnx(*args, **kwargs):
    """Lazy proxy so the core import doesn't depend on `onnx`."""
    from jax_ldt.export.onnx_export import to_onnx as _impl

    return _impl(*args, **kwargs)


__all__.append("to_onnx")


def to_discopt_decision_tree(*args, **kwargs):
    """Lazy proxy: convert a constant-leaf tree to `discopt.nn.tree.DecisionTree`."""
    from jax_ldt.export.discopt_adapter import to_discopt_decision_tree as _impl

    return _impl(*args, **kwargs)


def embed_in_discopt_model(*args, **kwargs):
    """Lazy proxy: hybrid embedding of an LMDT/HT into a discopt.Model."""
    from jax_ldt.export.discopt_adapter import embed_in_discopt_model as _impl

    return _impl(*args, **kwargs)


__all__.extend(["to_discopt_decision_tree", "embed_in_discopt_model"])
