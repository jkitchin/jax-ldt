"""Multi-target (T > 1) shape contracts across the public stack.

Coverage gap pinned by these tests (was G-22 in GAPS.md):
- LMDT and HT regressors fit and predict with shape (N, T)
- ONNX export round-trips for T > 1
- ConformalCalibrator produces (N, T) intervals
- ActiveLearner gates multi-target inputs at the boundary

These are smoke-level shape and consistency checks, not coverage
guarantees — those live in `test_conformal.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor
from jax_ldt.uncertainty import ConformalCalibrator


def _make_multitarget(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-1, 1, size=(n, 3))
    y0 = X[:, 0] + 0.5 * X[:, 1]
    y1 = X[:, 1] - 0.5 * X[:, 2]
    Y = np.stack([y0, y1], axis=1) + 0.05 * rng.standard_normal((n, 2))
    return X.astype(np.float64), Y.astype(np.float64)


@pytest.mark.parametrize("RegCls", [LinearTreeRegressor, HyperplaneTreeRegressor])
def test_regressor_predict_shape_multi_target(rng, RegCls) -> None:
    X, Y = _make_multitarget(rng)
    model = RegCls(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, Y)
    yh = np.asarray(model.predict(X))
    assert yh.shape == (X.shape[0], 2)
    assert model.n_targets_ == 2
    assert np.isfinite(yh).all()


def test_conformal_predict_interval_multi_target(rng) -> None:
    X, Y = _make_multitarget(rng, n=300)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=15).fit(
        X[:200], Y[:200]
    )
    cc = ConformalCalibrator(alpha=0.1, mondrian=False).calibrate(model, X[200:], Y[200:])
    lo, hi = cc.predict_interval(X[200:], model=model)
    lo_np, hi_np = np.asarray(lo), np.asarray(hi)
    assert lo_np.shape == (100, 2)
    assert hi_np.shape == (100, 2)
    # Halfwidths must be non-negative, intervals well-ordered.
    assert np.all(hi_np >= lo_np)


def test_active_learner_rejects_multi_target_y(rng) -> None:
    """ActiveLearner is single-output; multi-target y must surface a
    clear error rather than flatten silently inside an acquisition."""
    from jax_ldt.active_learning import ActiveLearner, ExpectedImprovement
    from jax_ldt.active_learning.batching import TopKBatchSelector

    X, Y = _make_multitarget(rng, n=80)
    learner = ActiveLearner(
        model_factory=lambda: LinearTreeRegressor(
            max_depth=2, max_bins=4, min_samples_leaf=10
        ),
        acquisition=ExpectedImprovement(direction="min"),
        batcher=TopKBatchSelector(),
        bounds=np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float64),
        seed=0,
    )
    with pytest.raises(ValueError, match="single-target"):
        learner.tell(X, Y)


def test_acquisition_sigma_rejects_multi_target_tree(rng) -> None:
    """`_sigma` (and therefore EI/PI/UCB) must fail fast when the model
    has n_targets > 1, rather than silently flatten σ.shape (N, T) into
    an (N·T,) array misaligned with μ."""
    from jax_ldt.active_learning.acquisitions import _sigma

    X, Y = _make_multitarget(rng, n=80)
    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=10).fit(X, Y)
    with pytest.raises(ValueError, match="single-output"):
        _sigma(model, X)


def test_onnx_round_trip_multi_target(rng) -> None:
    onnx = pytest.importorskip("onnx")  # noqa: F841
    ort = pytest.importorskip("onnxruntime")
    from jax_ldt.export import to_onnx

    X, Y = _make_multitarget(rng, n=200)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=15).fit(X, Y)
    proto = to_onnx(model)

    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"X": X.astype(np.float64)})[0]
    yh_jax = np.asarray(model.predict(X))
    assert out.shape == yh_jax.shape == (X.shape[0], 2)
    np.testing.assert_allclose(out, yh_jax, atol=1e-8, rtol=1e-8)
