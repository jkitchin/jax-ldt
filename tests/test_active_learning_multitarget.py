"""Multi-target active learning (issue #3).

The single-target constraint is a property of the *acquisition*, not of
the data: EI / PI score against a scalar best-y, while
`UncertaintySampler` / `MaxVariance` only read per-candidate uncertainty
and work for any number of targets. These tests pin

- an end-to-end exploration campaign over an (n, k) target,
- the acquisition-level capability flag and where the error fires,
- the per-target scaling + reduction that turns (N, T) sigma into one
  score per candidate,
- shape/target-count bookkeeping through tell / run / save / load.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import (
    ActiveLearner,
    ExpectedImprovement,
    GreedyMaxMinBatchSelector,
    LinearTreeRegressor,
    MaxVariance,
    ProbabilityOfImprovement,
    TopKBatchSelector,
    UncertaintySampler,
)
from jax_ldt.uncertainty import LinearPropagationUQ

BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)


def _model_factory():
    return LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=5)


def _targets(X: np.ndarray) -> np.ndarray:
    """Two targets on deliberately different magnitudes."""
    return np.column_stack(
        [X[:, 0] ** 2 + 0.3 * X[:, 1], 100.0 * np.sin(3.0 * X[:, 2])]
    )


def _learner(acquisition=None, **kw) -> ActiveLearner:
    return ActiveLearner(
        model_factory=_model_factory,
        acquisition=acquisition if acquisition is not None else UncertaintySampler(),
        batcher=kw.pop("batcher", GreedyMaxMinBatchSelector()),
        bounds=BOUNDS,
        batch_size=kw.pop("batch_size", 4),
        seed=kw.pop("seed", 0),
        **kw,
    )


def _fitted_multi_model(rng, n: int = 120):
    X = rng.uniform(0.0, 1.0, size=(n, 3))
    return X, LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=5).fit(
        X, _targets(X)
    )


# ---- capability flags ----


@pytest.mark.parametrize(
    ("acq", "expected"),
    [
        (UncertaintySampler(), False),
        (MaxVariance(), False),
        (ExpectedImprovement(), True),
        (ProbabilityOfImprovement(), True),
    ],
)
def test_requires_scalar_target_flag(acq, expected) -> None:
    assert acq.requires_scalar_target is expected


# ---- end-to-end campaign ----


def test_multi_target_campaign_runs_end_to_end(rng) -> None:
    """The exact scenario from issue #3: pure exploration over an (n, k)
    target must ask / tell / run without touching a scalarisation."""
    X = rng.uniform(0.0, 1.0, size=(60, 3))
    Y = _targets(X)

    loop = _learner()
    loop.tell(X[:40], Y[:40])
    assert loop.n_targets == 2
    assert loop.y_observed.shape == (40, 2)
    assert loop.model.n_targets_ == 2

    batch_id, X_batch = loop.ask()
    assert X_batch.shape == (4, 3)
    loop.tell(batch_id, _targets(X_batch))
    assert loop.y_observed.shape == (44, 2)

    loop.run(_targets, n_rounds=2)
    assert loop.y_observed.shape == (52, 2)
    assert [r.y.shape for r in loop.history] == [(4, 2)] * 3

    # The surrogate the campaign is building really is multi-output, and
    # so is its uncertainty.
    assert np.asarray(loop.model.predict(X[:5])).shape == (5, 2)
    assert np.asarray(LinearPropagationUQ().predict(loop.model, X[:5])).shape == (5, 2)


def test_ask_rejects_scalar_acquisition_swapped_in_later(rng) -> None:
    """`acquisition` is a public mutable attribute; a scalar-only one
    swapped in after multi-target data must be caught before scoring."""
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    loop = _learner()
    loop.tell(X, _targets(X))
    loop.acquisition = ExpectedImprovement(direction="min")
    with pytest.raises(ValueError, match="ExpectedImprovement.*single-target"):
        loop.ask()


def test_multi_target_save_load_round_trip(tmp_path, rng) -> None:
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    loop = _learner(batcher=TopKBatchSelector())
    loop.tell(X, _targets(X))
    batch_id, _ = loop.ask()
    path = tmp_path / "campaign.json"
    loop.save(path)

    loaded = ActiveLearner.load(
        path,
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
    )
    assert loaded.n_targets == 2
    assert loaded.y_observed.shape == (40, 2)
    np.testing.assert_allclose(loaded.y_observed, loop.y_observed)
    assert batch_id in loaded.pending
    # Refit on load must rebuild the *multi-output* tree.
    np.testing.assert_allclose(
        np.asarray(loaded.model.predict(X[:5])),
        np.asarray(loop.model.predict(X[:5])),
        atol=1e-12,
    )


def test_target_count_is_fixed_at_first_tell(rng) -> None:
    X = rng.uniform(0.0, 1.0, size=(30, 3))
    loop = _learner()
    loop.tell(X[:20], _targets(X[:20]))
    with pytest.raises(ValueError, match="target count is fixed"):
        loop.tell(X[20:], X[20:, 0])


def test_single_target_shapes_unchanged(rng) -> None:
    """(n, 1) still normalises to (n,) — single-target campaigns keep
    their historical y layout."""
    X = rng.uniform(0.0, 1.0, size=(30, 3))
    loop = _learner()
    loop.tell(X, (X[:, 0] ** 2)[:, None])
    assert loop.n_targets == 1
    assert loop.y_observed.shape == (30,)


def test_three_dimensional_y_rejected(rng) -> None:
    X = rng.uniform(0.0, 1.0, size=(10, 3))
    loop = _learner()
    with pytest.raises(ValueError, match=r"1-D \(n,\) or 2-D"):
        loop.tell(X, np.zeros((10, 2, 2)))


# ---- scaling + reduction ----


def test_scores_are_scaled_max_over_targets(rng) -> None:
    X, model = _fitted_multi_model(rng)
    sigma = np.asarray(LinearPropagationUQ().predict(model, X))
    assert sigma.shape == (X.shape[0], 2)

    scales = np.array([0.01, 10.0])
    acq = UncertaintySampler(target_scales=scales)
    np.testing.assert_allclose(
        acq.score(model, X), np.max(sigma / scales, axis=1), rtol=1e-12
    )

    acq_mean = UncertaintySampler(target_scales=scales, reduce="mean")
    np.testing.assert_allclose(
        acq_mean.score(model, X), np.mean(sigma / scales, axis=1), rtol=1e-12
    )

    # MaxVariance is the square of the same reduced sigma, so it stays
    # argmax-equivalent to UncertaintySampler.
    mv = MaxVariance(target_scales=scales)
    np.testing.assert_allclose(
        mv.score(model, X), np.max(sigma / scales, axis=1) ** 2, rtol=1e-12
    )
    assert int(np.argmax(mv.score(model, X))) == int(np.argmax(acq.score(model, X)))


def test_target_scales_change_which_candidate_wins(rng) -> None:
    """Scaling is the whole point: making one target's tolerance tight
    must move the batch toward that target's uncertain region."""
    X, model = _fitted_multi_model(rng)
    sigma = np.asarray(LinearPropagationUQ().predict(model, X))
    # Candidate rankings under "target 0 matters" vs "target 1 matters".
    top_0 = int(np.argmax(UncertaintySampler(target_scales=[1e-3, 1e3]).score(model, X)))
    top_1 = int(np.argmax(UncertaintySampler(target_scales=[1e3, 1e-3]).score(model, X)))
    assert top_0 == int(np.argmax(sigma[:, 0]))
    assert top_1 == int(np.argmax(sigma[:, 1]))


def test_unscaled_reduction_is_raw_units(rng) -> None:
    """Standalone, with no scales at all, the reduction is over raw
    per-target sigma (documented fallback)."""
    X, model = _fitted_multi_model(rng)
    sigma = np.asarray(LinearPropagationUQ().predict(model, X))
    np.testing.assert_allclose(
        UncertaintySampler().score(model, X), np.max(sigma, axis=1), rtol=1e-12
    )


def test_loop_supplies_observed_spread_as_fallback_scale(rng) -> None:
    """With no explicit target_scales, ActiveLearner hands the
    acquisition the observed per-target spread before scoring."""
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    Y = _targets(X)
    acq = UncertaintySampler()
    loop = _learner(acquisition=acq)
    loop.tell(X, Y)
    loop.ask()
    np.testing.assert_allclose(acq._observed_scales, np.std(Y, axis=0), rtol=1e-12)

    # And the scores actually use them.
    sigma = np.asarray(LinearPropagationUQ().predict(loop.model, X))
    np.testing.assert_allclose(
        acq.score(loop.model, X),
        np.max(sigma / np.std(Y, axis=0), axis=1),
        rtol=1e-12,
    )


def test_single_target_scores_are_unscaled_sigma(rng) -> None:
    """Single-target campaigns bypass scaling entirely: the score stays
    raw sigma even after the loop records observed scales."""
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    y = 100.0 * np.sin(3.0 * X[:, 2])
    acq = UncertaintySampler()
    loop = _learner(acquisition=acq)
    loop.tell(X, y)
    loop.ask()
    sigma = np.asarray(LinearPropagationUQ().predict(loop.model, X))
    assert sigma.ndim == 1
    np.testing.assert_allclose(acq.score(loop.model, X), sigma, rtol=1e-12)


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [1.0, 0.0], [1.0, -2.0], [np.nan, 1.0]])
def test_invalid_target_scales_rejected(rng, bad) -> None:
    X, model = _fitted_multi_model(rng, n=60)
    with pytest.raises(ValueError, match="target_scales"):
        UncertaintySampler(target_scales=bad).score(model, X)


def test_scalar_target_scale_is_broadcast(rng) -> None:
    X, model = _fitted_multi_model(rng, n=60)
    sigma = np.asarray(LinearPropagationUQ().predict(model, X))
    np.testing.assert_allclose(
        UncertaintySampler(target_scales=2.0).score(model, X),
        np.max(sigma / 2.0, axis=1),
        rtol=1e-12,
    )


def test_invalid_reduce_rejected() -> None:
    with pytest.raises(ValueError, match="reduce must be one of"):
        UncertaintySampler(reduce="median")
    with pytest.raises(ValueError, match="reduce must be one of"):
        MaxVariance(reduce="sum")


def test_update_best_rejects_multi_target_y() -> None:
    """EI/PI must not silently take a min over a flattened (n, k) y."""
    y = np.column_stack([np.arange(5.0), 100.0 + np.arange(5.0)])
    for acq in (ExpectedImprovement(), ProbabilityOfImprovement()):
        with pytest.raises(ValueError, match="single-target"):
            acq.update_best(y)
