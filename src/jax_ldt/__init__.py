"""jax-ldt: pure-JAX linear and hyperplane decision trees."""

from jax_ldt.linear_tree import LinearTreeRegressor
from jax_ldt.hyperplane_tree import HyperplaneTreeRegressor
from jax_ldt.hyperplanes import (
    build_transform_matrix,
    generate_planes_to_index,
    lift,
    symmetrize,
)
from jax_ldt._types import Tree, LeafUQ
from jax_ldt.uncertainty import (
    LinearPropagationUQ,
    QuadraticUQ,
    ConformalCalibrator,
    fit_with_conformal,
)
from jax_ldt.active_learning import (
    ActiveLearner,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UncertaintySampler,
    MaxVariance,
    TopKBatchSelector,
    DiverseBatchSelector,
    GreedyMaxMinBatchSelector,
)
from jax_ldt.export import to_json, from_json, to_onnx
from jax_ldt.tree_core import apply_tree, predict as predict_tree


def plot_tree_partition_2d(*args, **kwargs):
    """Lazy proxy for :func:`jax_ldt.viz.plot_tree_partition_2d`.

    Defers the matplotlib import until first call so the core package
    has no hard matplotlib dependency.
    """
    from jax_ldt.viz import plot_tree_partition_2d as _impl

    return _impl(*args, **kwargs)


def plot_calibration(*args, **kwargs):
    """Lazy proxy for :func:`jax_ldt.viz.plot_calibration`."""
    from jax_ldt.viz import plot_calibration as _impl

    return _impl(*args, **kwargs)


def __getattr__(name):  # pragma: no cover - exercised by SklearnLinearTreeRegressor test
    if name == "SklearnLinearTreeRegressor":
        from jax_ldt.sklearn_compat import SklearnLinearTreeRegressor

        return SklearnLinearTreeRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LinearTreeRegressor",
    "HyperplaneTreeRegressor",
    "Tree",
    "LeafUQ",
    "LinearPropagationUQ",
    "QuadraticUQ",
    "ConformalCalibrator",
    "fit_with_conformal",
    "to_json",
    "from_json",
    "to_onnx",
    "predict_tree",
    "apply_tree",
    "ActiveLearner",
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "UncertaintySampler",
    "MaxVariance",
    "TopKBatchSelector",
    "DiverseBatchSelector",
    "GreedyMaxMinBatchSelector",
    "plot_tree_partition_2d",
    "plot_calibration",
    "SklearnLinearTreeRegressor",
]

__version__ = "0.0.1"
