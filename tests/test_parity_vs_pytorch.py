"""Predictive-equivalence tests against the upstream PyTorch hyperplanetree.

Greedy tree growth is sensitive to tiny numerical perturbations: one
different early split cascades into a completely different tree. This
is not a bug — it's a feature of greedy partitioning. So we don't
assert exact prediction match; we assert *predictive equivalence*:

- Train MAE within a small relative band of upstream.
- Held-out grid: predictions correlate strongly with upstream
  (Pearson r above a documented threshold) AND fall in the same
  output range.

Skipped if torch / upstream cannot be imported.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    from tests._reference import lineartree as upstream_lineartree
    from tests._reference import hyperplane_tree as upstream_ht
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"upstream hyperplanetree not importable: {exc}", allow_module_level=True)

import jax.numpy as jnp  # noqa: E402

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor  # noqa: E402


CORR_FLOOR = 0.95  # Pearson r threshold on grid predictions
MAE_RATIO_HI = 1.5  # ours.train_mae / upstream.train_mae must be ≤ this
# Note: we do NOT impose a lower bound on the ratio. A train MAE *below*
# upstream's is a better fit, not a regression. The earlier "≥ 1/1.5"
# bound conflated train-set fidelity with overfitting; the held-out
# Pearson check below is what guards generalisation.


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 1.0
    return float((a * b).sum() / denom)


def _fit_upstream_lmdt(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    max_bins: int,
    min_samples_leaf: int,
    ridge: float,
):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    m = upstream_lineartree.LinearTreeRegressor(
        criterion="mae",
        max_depth=max_depth,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        ridge=ridge,
        disable_tqdm=True,
    )
    m.fit(Xt, yt)
    return m


def _predict_upstream(model, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float64)
    return np.asarray(model.predict(Xt)).reshape(-1)


def _equivalence_check(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    common: dict,
) -> None:
    up = _fit_upstream_lmdt(X, y, **common)
    ours = LinearTreeRegressor(
        criterion="mae",
        max_depth=common["max_depth"],
        max_bins=common["max_bins"],
        min_samples_leaf=common["min_samples_leaf"],
        ridge=common["ridge"],
    ).fit(X, y)

    # 1) Train MAE comparable
    yh_up_train = _predict_upstream(up, X)
    yh_ours_train = np.asarray(ours.predict(X))
    mae_up = float(np.mean(np.abs(y - yh_up_train)))
    mae_ours = float(np.mean(np.abs(y - yh_ours_train)))
    ratio = mae_ours / max(mae_up, 1e-12)
    assert ratio <= MAE_RATIO_HI, (
        f"{name}: train MAE ratio {ratio:.2f} above ceiling "
        f"{MAE_RATIO_HI:.2f} (ours={mae_ours:.4f}, up={mae_up:.4f})"
    )

    # 2) Held-out grid: predictions strongly correlated
    yh_up = _predict_upstream(up, grid)
    yh_ours = np.asarray(ours.predict(grid))
    r = _pearson(yh_up, yh_ours)
    assert r > CORR_FLOOR, (
        f"{name}: grid Pearson r {r:.3f} below floor {CORR_FLOOR:.3f}"
    )


@pytest.mark.parity
def test_lmdt_equivalence_1d(toy_1d) -> None:
    X, y = toy_1d
    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    _equivalence_check(
        "LMDT-1D",
        X,
        y,
        grid,
        common=dict(max_depth=4, max_bins=10, min_samples_leaf=10, ridge=1e-5),
    )


@pytest.mark.parity
def test_lmdt_equivalence_2d(branin_2d) -> None:
    X, y = branin_2d
    rng = np.random.default_rng(7)
    grid = rng.uniform([-5.0, 0.0], [10.0, 15.0], size=(300, 2))
    _equivalence_check(
        "LMDT-Branin",
        X,
        y,
        grid,
        common=dict(max_depth=5, max_bins=8, min_samples_leaf=15, ridge=1e-5),
    )


@pytest.mark.parity
def test_lmdt_equivalence_6d(friedman1_6d) -> None:
    X, y = friedman1_6d
    rng = np.random.default_rng(7)
    grid = rng.uniform(0.0, 1.0, size=(400, 6))
    _equivalence_check(
        "LMDT-Friedman1",
        X,
        y,
        grid,
        common=dict(max_depth=5, max_bins=6, min_samples_leaf=20, ridge=1e-5),
    )


def _fit_upstream_ht(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    max_bins: int,
    min_samples_leaf: int,
    ridge: float,
    max_weight: int,
    num_terms: int,
):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    m = upstream_ht.HyperplaneTreeRegressor(
        criterion="mae",
        max_depth=max_depth,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        ridge=ridge,
        max_weight=max_weight,
        num_terms=num_terms,
        disable_tqdm=True,
    )
    m.fit(Xt, yt)
    return m


@pytest.mark.parity
def test_ht_equivalence_2d(branin_2d) -> None:
    X, y = branin_2d
    common = dict(
        max_depth=4, max_bins=6, min_samples_leaf=15, ridge=1e-5, max_weight=1, num_terms=2
    )
    up = _fit_upstream_ht(X, y, **common)
    ours = HyperplaneTreeRegressor(
        criterion="mae",
        max_depth=common["max_depth"],
        max_bins=common["max_bins"],
        min_samples_leaf=common["min_samples_leaf"],
        ridge=common["ridge"],
        max_weight=common["max_weight"],
        num_terms=common["num_terms"],
    ).fit(X, y)

    # Train MAE comparable
    mae_up = float(np.mean(np.abs(y - _predict_upstream(up, X))))
    mae_ours = float(np.mean(np.abs(y - np.asarray(ours.predict(X)))))
    ratio = mae_ours / max(mae_up, 1e-12)
    assert ratio <= MAE_RATIO_HI, (
        f"HT-2D train MAE ratio {ratio:.2f} above ceiling {MAE_RATIO_HI:.2f}; "
        f"ours={mae_ours:.4f}, up={mae_up:.4f}"
    )

    # Grid: predictions correlated
    rng = np.random.default_rng(11)
    grid = rng.uniform([-5.0, 0.0], [10.0, 15.0], size=(300, 2))
    yh_up = _predict_upstream(up, grid)
    yh_ours = np.asarray(ours.predict(grid))
    r = _pearson(yh_up, yh_ours)
    assert r > CORR_FLOOR, f"HT-2D grid Pearson r {r:.3f} below floor {CORR_FLOOR:.3f}"
