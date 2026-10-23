"""Fit-cost scaling of the hyperplane tree (issue #2).

`HyperplaneTreeRegressor.fit` used to evaluate every candidate split with
a dense ``O(n_aug**3)`` ridge solve in the *lifted* space, so cost grew
as ``O(n_lifted**4)`` — at the defaults that is ``n_features**8``, and a
16-feature problem took ~112 s where a 4-feature one took ~3 s.

The kernel now solves each candidate in the ``m = n_features_in + 1``
dimensional space it actually spans (see design note 5 in
`tree_core`), which makes candidate cost independent of the lifted-
feature count. These tests pin both halves of that claim:

- **exactness** — the reduced-space losses equal the dense ridge fit's;
- **cost** — the compiled kernel's flop count does not move when the
  lifted width does.

plus the up-front sizing tools (`count_lifted_features`,
`max_hyperplanes`, the size warning) that keep an intractable
configuration legible instead of silently hanging.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, count_lifted_features
from jax_ldt.hyperplanes import (
    LIFTED_FEATURE_WARN_THRESHOLD,
    build_transform_matrix,
)
from jax_ldt.linear_regression import add_bias
from jax_ldt.tree_core import (
    _WEIGHTED_LOSS_FNS,
    _compute_thresholds,
    _evaluate_splits,
    _evaluate_splits_chunked,
    _fit_one_side,
    reduced_split_basis,
)


def _transform(X: np.ndarray, **kw) -> jnp.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build_transform_matrix(jnp.asarray(X), **kw)


def _setup(n_features=4, n=90, max_weight=1, num_terms=2, drop_cols=(), seed=0):
    """Build everything both the kernel and a dense reference fit need."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, n_features))
    y = (np.sin(X.sum(axis=1)) + 0.1 * rng.standard_normal(n))[:, None]
    A = _transform(X, num_terms=num_terms, max_weight=max_weight)
    n_aug = int(A.shape[1]) + 1

    lin_cols = np.ones(n_aug, dtype=bool)
    for c in drop_cols:
        lin_cols[c] = False
    lin_cols[0] = True  # bias always retained

    X_t = jnp.asarray(X) @ A
    X_fit = add_bias(X_t) * jnp.asarray(lin_cols, dtype=X_t.dtype)[None, :]
    Z, BBt, ZK = reduced_split_basis(jnp.asarray(X), A, lin_cols)
    return {
        "X": X, "y": jnp.asarray(y), "A": A, "X_t": X_t, "X_fit": X_fit,
        "Z": Z, "BBt": BBt, "ZK": ZK, "lin_cols": lin_cols, "n_aug": n_aug,
    }


def _dense_losses(loss_name, X_fit, y, node_mask, X_split, thresholds,
                  ridge, min_samples_leaf):
    """Reference (K, B) candidate losses via dense n_aug-space ridge fits."""
    fn = _WEIGHTED_LOSS_FNS[loss_name]
    nm = np.asarray(node_mask)
    thr = np.asarray(thresholds)
    xs = np.asarray(X_split)
    B, K = thr.shape
    out = np.full((K, B), np.inf)
    for c in range(K):
        for b in range(B):
            below = xs[:, c] <= thr[b, c]
            mb = nm * below.astype(float)
            ma = nm * (~below).astype(float)
            if mb.sum() < min_samples_leaf or ma.sum() < min_samples_leaf:
                continue
            th_b = _fit_one_side(X_fit, y, jnp.asarray(mb), ridge)
            th_a = _fit_one_side(X_fit, y, jnp.asarray(ma), ridge)
            yh = (X_fit @ th_b) * mb[:, None] + (X_fit @ th_a) * ma[:, None]
            out[c, b] = float(jnp.sum(fn(y, yh, jnp.asarray(nm))))
    return out


# ---------------------------------------------------------------------------
# Exactness: reduced-space solve == dense solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_name", ["mae", "mse", "rmse", "max_abs"])
@pytest.mark.parametrize("drop_cols", [(), (2, 5, 9)])
def test_reduced_kernel_matches_dense_ridge(loss_name, drop_cols) -> None:
    """Kernel losses must match an explicit n_aug-space ridge fit.

    The push-through identity the kernel relies on is exact in exact
    arithmetic, so the only allowed difference is round-off. `drop_cols`
    exercises the `linear_features` mask, which makes ``B Bᵀ`` singular —
    the identity has to survive that too.
    """
    d = _setup(drop_cols=drop_cols)
    mask = jnp.ones(d["X"].shape[0])
    X_split = d["X_t"][:, :6]
    thresholds = _compute_thresholds(np.asarray(X_split), max_bins=8)

    overall, _, _ = _evaluate_splits(
        d["Z"], d["ZK"], d["BBt"], d["y"], mask, X_split, thresholds,
        1e-5, 5.0, loss_name,
    )
    overall = np.asarray(overall)
    dense = _dense_losses(
        loss_name, d["X_fit"], d["y"], mask, X_split, thresholds, 1e-5, 5.0
    )

    finite = np.isfinite(dense)
    assert finite.sum() > 10, "expected most candidates to be valid"
    np.testing.assert_allclose(overall[finite], dense[finite], rtol=1e-9)
    # Invalid candidates agree on being invalid.
    assert np.all(np.isinf(overall[~finite]))


def test_reduced_kernel_matches_dense_on_wide_lift() -> None:
    """Same equivalence where the lift is much wider than the input.

    ``num_terms=3, max_weight=2`` on 4 features lifts to 152 columns, so
    the dense fit is a 153-dimensional solve and the kernel's is a
    5-dimensional one. They must still agree.
    """
    d = _setup(n_features=4, num_terms=3, max_weight=2)
    assert d["n_aug"] > 100
    mask = jnp.ones(d["X"].shape[0])
    X_split = d["X_t"][:, :5]
    thresholds = _compute_thresholds(np.asarray(X_split), max_bins=6)

    overall, _, _ = _evaluate_splits(
        d["Z"], d["ZK"], d["BBt"], d["y"], mask, X_split, thresholds,
        1e-5, 5.0, "mae",
    )
    dense = _dense_losses(
        "mae", d["X_fit"], d["y"], mask, X_split, thresholds, 1e-5, 5.0
    )
    finite = np.isfinite(dense)
    assert finite.any()
    np.testing.assert_allclose(np.asarray(overall)[finite], dense[finite], rtol=1e-8)


# ---------------------------------------------------------------------------
# Cost: independent of the lifted-feature count
# ---------------------------------------------------------------------------


def _kernel_flops(d, K=8, n_bins=9):
    X_split = d["X_t"][:, :K]
    thresholds = _compute_thresholds(np.asarray(X_split), max_bins=n_bins + 1)
    mask = jnp.ones(d["X"].shape[0])
    compiled = _evaluate_splits.lower(
        d["Z"], d["ZK"], d["BBt"], d["y"], mask, X_split, thresholds,
        1e-5, 5.0, "mae",
    ).compile()
    cost = compiled.cost_analysis()
    if isinstance(cost, list):  # older jaxlib returned a per-module list
        cost = cost[0]
    if not isinstance(cost, dict) or "flops" not in cost:
        pytest.skip("XLA cost analysis unavailable on this backend")
    return float(cost["flops"])


def test_kernel_cost_is_independent_of_lifted_width() -> None:
    """The regression guard for issue #2.

    Hold the input dimension and the candidate count fixed and widen only
    the lift. The compiled kernel must do exactly the same work — that is
    what stops fit cost from growing with the lifted-feature count. The
    pre-fix kernel solved an ``n_aug × n_aug`` system per candidate, so
    its flop count grew cubically across these three configurations.
    """
    flops = []
    widths = []
    for max_weight in (1, 2, 3):
        d = _setup(n_features=4, max_weight=max_weight)
        widths.append(d["n_aug"])
        flops.append(_kernel_flops(d))

    assert widths[-1] > 5 * widths[0], f"lift did not widen enough: {widths}"
    assert len(set(flops)) == 1, f"kernel cost moved with lifted width: {list(zip(widths, flops))}"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_cols", [1, 2, 5, 7, 100])
def test_chunked_losses_match_single_call(chunk_cols) -> None:
    """Column chunking (with its padded tail) must not change any loss."""
    d = _setup(n_features=4)
    mask = jnp.ones(d["X"].shape[0])
    X_split = d["X_t"][:, :7]
    thresholds = _compute_thresholds(np.asarray(X_split), max_bins=8)

    reference, _, _ = _evaluate_splits(
        d["Z"], d["ZK"], d["BBt"], d["y"], mask, X_split, thresholds,
        1e-5, 5.0, "mae",
    )
    chunked = _evaluate_splits_chunked(
        d["Z"], d["ZK"], d["BBt"], d["y"], mask, X_split, thresholds,
        1e-5, 5.0, "mae", chunk_cols,
    )
    reference = np.asarray(reference)
    assert chunked.shape == reference.shape == (7, thresholds.shape[0])
    # Chunk width changes the batch shape XLA reduces over, so agreement
    # is to round-off, not bit-for-bit.
    finite = np.isfinite(reference)
    assert finite.any()
    np.testing.assert_allclose(chunked[finite], reference[finite], rtol=1e-12)
    np.testing.assert_array_equal(np.isinf(chunked), np.isinf(reference))


def test_chunking_does_not_change_the_fitted_tree(monkeypatch) -> None:
    """A tiny memory budget forces many chunks; the tree must be identical."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(80, 4))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1] * X[:, 2]

    baseline = HyperplaneTreeRegressor(max_depth=3).fit(X, y)

    import jax_ldt.tree_core as tc

    monkeypatch.setattr(tc, "SPLIT_MEMORY_BUDGET_BYTES", 1)
    chunked = HyperplaneTreeRegressor(max_depth=3).fit(X, y)

    np.testing.assert_array_equal(
        np.asarray(baseline.tree_.feature), np.asarray(chunked.tree_.feature)
    )
    np.testing.assert_allclose(
        np.asarray(baseline.tree_.threshold), np.asarray(chunked.tree_.threshold)
    )
    np.testing.assert_allclose(baseline.predict(X), chunked.predict(X), rtol=1e-12)


# ---------------------------------------------------------------------------
# Sizing a configuration before fitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_features,num_terms,max_weight",
    [(3, 2, 1), (4, 2, 1), (6, 2, 1), (4, 2, 2), (4, 3, 1), (5, 3, 2)],
)
def test_count_lifted_features_matches_transform(n_features, num_terms, max_weight) -> None:
    """The advertised count is the width the fit actually uses."""
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(6, n_features))
    A = _transform(X, num_terms=num_terms, max_weight=max_weight)
    assert count_lifted_features(
        n_features, num_terms=num_terms, max_weight=max_weight
    ) == int(A.shape[1])


@pytest.mark.parametrize("n_features", [2, 3, 4, 7, 12])
def test_default_lift_is_n_features_squared(n_features) -> None:
    """Documented closed form for the default (num_terms=2, max_weight=1)."""
    assert count_lifted_features(n_features) == n_features**2


def test_count_lifted_features_is_silent() -> None:
    """Sizing a huge configuration must not itself emit the size warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        n = count_lifted_features(8, num_terms=4, max_weight=2)
    assert n > LIFTED_FEATURE_WARN_THRESHOLD


def test_large_configuration_warns() -> None:
    """An expensive lift announces itself rather than hanging silently."""
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(6, 8))
    with pytest.warns(UserWarning, match="lifts to"):
        build_transform_matrix(jnp.asarray(X), num_terms=4, max_weight=2)


def test_no_warning_for_ordinary_configuration() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A = build_transform_matrix(jnp.asarray(X))
    assert int(A.shape[1]) == 36


# ---------------------------------------------------------------------------
# max_hyperplanes
# ---------------------------------------------------------------------------


def test_max_hyperplanes_caps_the_lift() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 6))
    uncapped = int(_transform(X, num_terms=3, max_weight=2).shape[1])
    with pytest.warns(UserWarning, match="capped the oblique directions"):
        capped = build_transform_matrix(
            jnp.asarray(X), num_terms=3, max_weight=2, max_hyperplanes=50
        )
    # Identity block is always kept on top of the capped oblique rows.
    assert int(capped.shape[1]) <= 6 + 50 < uncapped
    assert int(capped.shape[1]) > 6


def test_max_hyperplanes_is_deterministic() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 6))
    kw = dict(num_terms=3, max_weight=2, max_hyperplanes=40)
    a = _transform(X, **kw)
    b = _transform(X, **kw)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_max_hyperplanes_no_op_below_cap() -> None:
    """A cap above the enumeration size changes nothing and stays quiet."""
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 4))
    plain = _transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped = build_transform_matrix(jnp.asarray(X), max_hyperplanes=10_000)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(capped))


def test_max_hyperplanes_zero_keeps_identity_only() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 4))
    with pytest.warns(UserWarning, match="capped the oblique directions"):
        A = build_transform_matrix(jnp.asarray(X), max_hyperplanes=0)
    assert A.shape == (4, 4)


def test_max_hyperplanes_rejects_negative() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 4))
    with pytest.raises(ValueError, match="max_hyperplanes"):
        build_transform_matrix(jnp.asarray(X), max_hyperplanes=-1)


def test_regressor_forwards_max_hyperplanes() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(60, 5))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(60)
    with pytest.warns(UserWarning, match="capped the oblique directions"):
        model = HyperplaneTreeRegressor(
            max_depth=2, num_terms=3, max_weight=2, max_hyperplanes=20
        ).fit(X, y)
    assert model.transform_matrix_.shape[1] <= 5 + 20
    assert np.all(np.isfinite(np.asarray(model.predict(X))))


# ---------------------------------------------------------------------------
# End-to-end: the shapes from the issue actually finish
# ---------------------------------------------------------------------------


def _fit_seconds(n_features: int, **kw) -> float:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(120, n_features))
    y = np.random.default_rng(1).uniform(0.0, 1.0, size=120)
    start = time.perf_counter()
    HyperplaneTreeRegressor(criterion="mae", max_depth=4, **kw).fit(X, y)
    return time.perf_counter() - start


def test_wide_fit_completes_quickly() -> None:
    """A 16-feature fit (256 lifted columns) took ~112 s before the fix.

    The bound is deliberately loose — it is a "did the quartic term come
    back" tripwire, not a benchmark. Measured ~3 s on a 4-core CPU
    runner, so a 60 s ceiling tolerates a ~20x slower machine while still
    failing outright against the pre-fix kernel.
    """
    jax.block_until_ready(jnp.zeros(1))  # pay any backend warmup first
    assert _fit_seconds(16) < 60.0


@pytest.mark.perf
def test_issue_2_feature_count_cliff() -> None:
    """Issue #2's acceptance criterion, as a wall-clock ratio.

    "`HyperplaneTreeRegressor` fits the 4-feature repro in time comparable
    to the 3-feature case." Before the fix the 4-feature fit never
    finished; a 3x band on the 3-feature time is comfortably met now.
    """
    _fit_seconds(3)  # warm the JIT cache
    t3 = _fit_seconds(3)
    t4 = _fit_seconds(4)
    assert t4 < max(3.0 * t3, 10.0), f"3 features: {t3:.1f}s, 4 features: {t4:.1f}s"
