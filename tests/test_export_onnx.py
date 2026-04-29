"""ONNX export + onnxruntime round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import jax.numpy as jnp  # noqa: E402

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor, to_onnx  # noqa: E402


def _run_onnx(model_proto, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_proto.SerializeToString())
    out = sess.run(None, {"X": X.astype(np.float64)})
    return np.asarray(out[0])


# Each entry: (id, factory, n_test_rows, persist_to_disk).
# The previous version copy-pasted this body for LMDT and HT separately;
# parametrizing here means new regressor classes only need a row here
# rather than a fresh test function.
_ROUND_TRIP_CASES = [
    pytest.param(
        lambda: LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=15),
        50,
        True,
        id="LMDT",
    ),
    pytest.param(
        lambda: HyperplaneTreeRegressor(
            max_depth=4, max_bins=6, min_samples_leaf=15, max_weight=1, num_terms=2
        ),
        30,
        False,
        id="HT",
    ),
]


@pytest.mark.onnx
@pytest.mark.parametrize("factory, n_test, persist", _ROUND_TRIP_CASES)
def test_onnx_round_trip(branin_2d, tmp_path, factory, n_test, persist) -> None:
    """LMDT and HT both feed the same ONNX export path; both must
    round-trip predictions to within float64-rounding tolerance.

    Tolerance ``1e-8`` is empirical: the ONNX graph compounds float64
    rounding through MatMul + Gather + ReduceSum, which can drift a few
    ULPs even on identical hardware.
    """
    X, y = branin_2d
    model = factory().fit(X, y)

    out_path = tmp_path / "tree.onnx" if persist else None
    model_proto = to_onnx(model, out_path) if persist else to_onnx(model)
    if persist:
        assert out_path.exists()

    X_test = X[:n_test]
    yh_jax = np.asarray(model.predict(X_test))
    yh_onnx = _run_onnx(model_proto, X_test).reshape(-1)
    np.testing.assert_allclose(yh_jax, yh_onnx, atol=1e-8)


@pytest.mark.onnx
def test_ht_onnx_round_trip_deeper_tree(friedman1_6d) -> None:
    """G-14: deeper hyperplane tree, larger test batch, tighter tolerance.

    Documents that even with depth-8 routing the round-trip stays within
    1e-7 — well below typical surrogate-modelling needs.
    """
    X, y = friedman1_6d
    model = HyperplaneTreeRegressor(
        max_depth=8,
        max_bins=10,
        min_samples_leaf=10,
        max_weight=1,
        num_terms=2,
    ).fit(X, y)
    model_proto = to_onnx(model)

    X_test = X[:100]
    yh_jax = np.asarray(model.predict(X_test))
    yh_onnx = _run_onnx(model_proto, X_test).reshape(-1)
    np.testing.assert_allclose(yh_jax, yh_onnx, atol=1e-7)


@pytest.mark.onnx
def test_onnx_constant_target(rng) -> None:
    """Edge case: tree with a single leaf (constant prediction)."""
    X = rng.uniform(-1, 1, size=(50, 2))
    y = np.full(50, 3.5)
    model = LinearTreeRegressor(max_depth=2, max_bins=3, min_samples_leaf=10).fit(X, y)
    model_proto = to_onnx(model)
    yh_onnx = _run_onnx(model_proto, X).reshape(-1)
    np.testing.assert_allclose(yh_onnx, 3.5, atol=1e-3)


@pytest.mark.onnx
def test_onnx_multi_target(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(120, 2)).astype(np.float64)
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]]).astype(np.float64)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    model_proto = to_onnx(model)
    yh_jax = np.asarray(model.predict(X[:25]))
    yh_onnx = _run_onnx(model_proto, X[:25])
    assert yh_onnx.shape == (25, 2)
    np.testing.assert_allclose(yh_jax, yh_onnx, atol=1e-8)
