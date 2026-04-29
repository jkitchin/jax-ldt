"""Wall-clock performance smoke tests.

These exist so the ``perf`` marker referenced by ``pyproject.toml`` and
``docs/benchmarks.md`` actually has something to run. They are NOT
microbenchmarks — they check that fit / predict / active-learning
performance has not regressed catastrophically (e.g., a refactor that
re-introduces per-node retracing or per-leaf host syncs would blow
through these bounds by an order of magnitude).

Marked ``perf`` so a default ``pytest`` invocation skips them; opt in
with ``pytest -m perf``. The thresholds are deliberately loose to
survive cold JIT caches and slower CI hardware; tighten them if you
adopt ``pytest-benchmark`` for proper statistics.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from jax_ldt import (
    ActiveLearner,
    ExpectedImprovement,
    LinearTreeRegressor,
    TopKBatchSelector,
)


pytestmark = pytest.mark.perf


def _time_call(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def test_lmdt_fit_under_budget(rng) -> None:
    """LMDT fit on a 5k × 6 dataset should complete inside a generous
    wall-clock budget. Previously a refactor that triggered per-node
    JIT retracing would have blown this by 5-10×."""
    n = 5000
    X = rng.uniform(-2, 2, size=(n, 6)).astype(np.float64)
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    ).astype(np.float64)

    # Warm up the JIT cache so the timing reflects steady-state cost.
    LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=10).fit(
        X[:200], y[:200]
    )

    elapsed, model = _time_call(
        LinearTreeRegressor(max_depth=5, max_bins=8, min_samples_leaf=20).fit,
        X,
        y,
    )
    # 30 s on a slow CI runner is still 5-10× a "normal" fit; a
    # regression that loses JIT caching surfaces as 60+ s.
    assert elapsed < 30.0, f"LMDT fit slow: {elapsed:.2f}s"
    # Sanity: a non-trivial tree with an actually-fitted model.
    assert int(np.asarray(model.tree_.is_leaf).sum()) > 1


def test_lmdt_predict_throughput(rng) -> None:
    """Predict on 100k points should be fast once the JIT cache is
    populated. Acts as a regression guard on the routing/leaf-gather
    path."""
    n_train = 1000
    n_predict = 100_000
    X = rng.uniform(-1, 1, size=(n_train, 4)).astype(np.float64)
    y = X.sum(axis=1).astype(np.float64)
    model = LinearTreeRegressor(max_depth=4, max_bins=6, min_samples_leaf=15).fit(X, y)

    X_test = rng.uniform(-1, 1, size=(n_predict, 4)).astype(np.float64)
    # Warm-up to compile the routing kernel for this test-batch shape.
    np.asarray(model.predict(X_test[:1024])).block_until_ready() if hasattr(
        np.asarray(model.predict(X_test[:1024])), "block_until_ready"
    ) else None

    elapsed, yh = _time_call(model.predict, X_test)
    yh_np = np.asarray(yh)
    assert yh_np.shape == (n_predict,)
    # 5 s on 100k points is still ~50 µs / row including all framework
    # overhead. Catastrophic regressions land in the 30+ s range.
    assert elapsed < 5.0, f"predict slow: {elapsed:.2f}s"


def test_active_learning_loop_under_budget(rng) -> None:
    """A 5-round AL loop with batch_size=8 and a non-trivial model
    factory should complete in seconds, not minutes. Catches accidental
    per-round full-recompilation."""
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)

    def factory():
        return LinearTreeRegressor(max_depth=4, max_bins=6, min_samples_leaf=5)

    loop = ActiveLearner(
        model_factory=factory,
        acquisition=ExpectedImprovement(direction="min"),
        batcher=TopKBatchSelector(),
        bounds=bounds,
        batch_size=8,
        seed=42,
    )
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop.tell(X0, y0)

    def f(X):
        return (X ** 2).sum(axis=1)

    elapsed, _ = _time_call(loop.run, run_experiment=f, n_rounds=5)
    assert elapsed < 60.0, f"AL loop slow: {elapsed:.2f}s"
    # Sanity: every round added 8 rows.
    assert loop.X_observed.shape[0] == 20 + 5 * 8
