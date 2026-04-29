"""Smoke tests for HyperplaneTreeRegressor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt import HyperplaneTreeRegressor


def test_fit_predict_smoke_2d(branin_2d) -> None:
    X, y = branin_2d
    model = HyperplaneTreeRegressor(
        max_depth=5, max_bins=6, min_samples_leaf=15, max_weight=1, num_terms=2
    )
    model.fit(X, y)
    yh = model.predict(X)
    assert yh.shape == (X.shape[0],)
    # Depth-5 HT on Branin reliably hits R² ≈ 0.9+; the band catches
    # both real regressions (lower) and over-fitting (upper).
    ss_res = float(jnp.sum((y - yh) ** 2))
    ss_tot = float(jnp.sum((y - jnp.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert 0.85 <= r2 <= 0.999, f"HT R^2 outside expected band: {r2:.3f}"


def test_ht_outperforms_or_matches_lmdt_on_diagonal_function(rng) -> None:
    """A deliberately diagonal target (y = x0 + x1) should be cleanly fit
    by HT with a single oblique split, but require many axis-aligned LMDT splits.
    """
    from jax_ldt import LinearTreeRegressor

    X = rng.uniform(-2, 2, size=(300, 2))
    y = X[:, 0] + X[:, 1] + 0.05 * rng.standard_normal(300)

    common = dict(max_depth=4, max_bins=6, min_samples_leaf=15)
    lmdt = LinearTreeRegressor(**common, ridge=1e-5).fit(X, y)
    ht = HyperplaneTreeRegressor(**common, ridge=1e-5, max_weight=1, num_terms=2).fit(X, y)

    mae_lmdt = float(jnp.mean(jnp.abs(lmdt.predict(X) - y)))
    mae_ht = float(jnp.mean(jnp.abs(ht.predict(X) - y)))
    # Both should be small; HT should not be obviously worse.
    assert mae_ht <= 1.5 * mae_lmdt + 0.05, (
        f"HT MAE {mae_ht:.4f} much worse than LMDT MAE {mae_lmdt:.4f}"
    )


def test_transform_matrix_first_block_is_identity(branin_2d) -> None:
    X, y = branin_2d
    ht = HyperplaneTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    A = ht.transform_matrix_
    n_in = X.shape[1]
    np.testing.assert_allclose(np.asarray(A[:, :n_in]), np.eye(n_in), atol=1e-6)


def test_split_features_restricts_oblique_set(branin_2d) -> None:
    """`split_features=(0, 1)` (only the two original axis-aligned columns
    of the transformed space) should yield a tree whose `tree_.feature`
    array only contains values from {0, 1, -1}.
    """
    X, y = branin_2d
    ht = HyperplaneTreeRegressor(
        max_depth=4,
        max_bins=6,
        min_samples_leaf=20,
        split_features=(0, 1),
        max_weight=1,
        num_terms=2,
    ).fit(X, y)
    feats = np.asarray(ht.tree_.feature).tolist()
    allowed = {0, 1, -1}
    bad = [f for f in feats if f not in allowed]
    assert not bad, f"unexpected split feature ids: {bad}"


def test_apply_returns_leaf_ids(branin_2d) -> None:
    X, y = branin_2d
    model = HyperplaneTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=20, max_weight=1, num_terms=2
    ).fit(X, y)
    leaves = model.apply(X)
    assert leaves.shape == (X.shape[0],)
    is_leaf = np.asarray(model.tree_.is_leaf)
    assert is_leaf[np.asarray(leaves)].all()
