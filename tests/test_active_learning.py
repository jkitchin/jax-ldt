"""Active learning loop tests."""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import (
    ActiveLearner,
    DiverseBatchSelector,
    ExpectedImprovement,
    GreedyMaxMinBatchSelector,
    LinearTreeRegressor,
    TopKBatchSelector,
    UncertaintySampler,
)


def _model_factory():
    return LinearTreeRegressor(max_depth=4, max_bins=6, min_samples_leaf=5)


def _bounds_2d() -> np.ndarray:
    return np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)


def test_ask_before_tell_raises() -> None:
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=_bounds_2d(),
        batch_size=4,
    )
    with pytest.raises(RuntimeError):
        loop.ask()


def test_basic_loop_uncertainty(rng) -> None:
    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=bounds,
        batch_size=4,
        seed=42,
    )
    # Seed with 20 random points
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop.tell(X0, y0)

    bid, X_batch = loop.ask()
    assert X_batch.shape == (4, 2)
    assert bid in loop.pending

    y_new = (X_batch ** 2).sum(axis=1)
    loop.tell(bid, y_new)
    assert loop.X_observed.shape == (24, 2)
    assert bid not in loop.pending


def test_partial_tell_keeps_pending(rng) -> None:
    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=bounds,
        batch_size=4,
    )
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(15, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop.tell(X0, y0)
    bid, X_batch = loop.ask()

    # Only respond to indices [0, 2]
    indices = [0, 2]
    y_partial = (X_batch[indices] ** 2).sum(axis=1)
    loop.tell(bid, y_partial, indices=indices)

    # Two rows added; batch still pending; remaining indices unanswered
    assert loop.X_observed.shape == (17, 2)
    assert bid in loop.pending
    assert sorted(loop.pending[bid].answered_indices) == [0, 2]

    # Finish the batch
    rest = [1, 3]
    y_rest = (X_batch[rest] ** 2).sum(axis=1)
    loop.tell(bid, y_rest, indices=rest)
    assert loop.X_observed.shape == (19, 2)
    assert bid not in loop.pending


def test_ei_finds_minimum_on_negated_quadratic() -> None:
    """Sanity: EI on a simple bowl should drive samples toward the minimum.

    f(x) = (x - target)^2; minimum at x = target. After several rounds,
    the best observed value should be < a documented threshold.
    """
    target = np.array([0.5, -0.7])

    def f(X: np.ndarray) -> np.ndarray:
        return ((X - target) ** 2).sum(axis=1)

    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=ExpectedImprovement(direction="min"),
        batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
        bounds=bounds,
        batch_size=6,
        seed=11,
    )
    rng = np.random.default_rng(11)
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(8, 2))
    loop.tell(X0, f(X0))

    loop.run(run_experiment=f, n_rounds=10)

    best = float(np.min(loop.y_observed))
    # The minimum is 0; with 8 + 60 evals and EI, we should find < 0.05.
    assert best < 0.05, f"AL did not converge: best={best:.4f}"


def test_save_and_load_round_trip(tmp_path, rng) -> None:
    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=bounds,
        batch_size=3,
        seed=7,
    )
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(12, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop.tell(X0, y0)
    bid, _ = loop.ask()

    path = tmp_path / "campaign.json"
    loop.save(path)

    loaded = ActiveLearner.load(
        path,
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
    )
    np.testing.assert_allclose(loaded.X_observed, loop.X_observed)
    np.testing.assert_allclose(loaded.y_observed, loop.y_observed)
    assert bid in loaded.pending
    # The loaded loop must auto-refit and produce identical predictions
    # to the original — the previous "is not None" only checked that
    # _something_ was assigned. Pin equality on a small probe set so a
    # regression where the wrong tree is rebuilt would surface.
    probe = X0[:5]
    yh_orig = np.asarray(loop.model.predict(probe))
    yh_loaded = np.asarray(loaded.model.predict(probe))
    np.testing.assert_allclose(yh_loaded, yh_orig, atol=1e-12)


def test_multi_target_y_rejected(rng) -> None:
    """G-9: tell() with multi-target y must raise a clear error, not silently flatten."""
    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=bounds,
        batch_size=4,
    )
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, 2))
    y0 = np.column_stack([X0[:, 0], X0[:, 1]])  # shape (10, 2): 2 targets
    with pytest.raises(ValueError, match="single-target"):
        loop.tell(X0, y0)


def test_save_load_preserves_candidate_stream(tmp_path, rng) -> None:
    """G-2: ask() after save/load yields the same batch as ask() without save."""
    bounds = _bounds_2d()

    def make_loop():
        return ActiveLearner(
            model_factory=_model_factory,
            acquisition=UncertaintySampler(),
            batcher=TopKBatchSelector(),
            bounds=bounds,
            batch_size=4,
            seed=12345,
        )

    loop_a = make_loop()
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(15, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop_a.tell(X0, y0)
    bid_a, X_batch_a = loop_a.ask()

    # Same setup, save before ask, load, then ask.
    loop_b = make_loop()
    loop_b.tell(X0, y0)
    path = tmp_path / "campaign.json"
    loop_b.save(path)
    loop_c = ActiveLearner.load(
        path,
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
    )
    bid_c, X_batch_c = loop_c.ask()

    np.testing.assert_allclose(
        X_batch_a, X_batch_c,
        err_msg="ask() after save/load should reproduce the same candidates",
    )


def test_diverse_batch_returns_distinct_points(rng) -> None:
    bounds = _bounds_2d()
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=DiverseBatchSelector(pool_factor=8),
        bounds=bounds,
        batch_size=5,
        seed=3,
    )
    X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
    y0 = (X0 ** 2).sum(axis=1)
    loop.tell(X0, y0)
    bid, X_batch = loop.ask()
    # All points should be distinct
    assert X_batch.shape == (5, 2)
    pairwise = np.linalg.norm(
        X_batch[:, None, :] - X_batch[None, :, :], axis=-1
    )
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() > 1e-6


def test_max_variance_is_distinct_from_uncertainty_sampler() -> None:
    """`MaxVariance` returns σ² (variance); `UncertaintySampler` returns
    σ. They are distinct classes with the same argmax behaviour but
    different score values — the previous "alias" arrangement made the
    name misleading."""
    from jax_ldt import MaxVariance, UncertaintySampler
    from jax_ldt.active_learning import (
        MaxVariance as MaxVariance2,
        UncertaintySampler as UncertaintySampler2,
    )

    # Re-exports are consistent across import paths.
    assert MaxVariance is MaxVariance2
    assert UncertaintySampler is UncertaintySampler2
    # And the two classes are no longer the same object.
    assert MaxVariance is not UncertaintySampler
    assert not isinstance(MaxVariance(), UncertaintySampler)
