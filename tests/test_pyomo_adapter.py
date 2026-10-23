"""Pyomo adapter tests. Skipped if pyomo is not installed; solve tests
additionally need the HiGHS solver (`pip install highspy`)."""

from __future__ import annotations

import numpy as np
import pytest

pyo = pytest.importorskip("pyomo.environ")

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor  # noqa: E402
from jax_ldt.export import embed_in_pyomo_model  # noqa: E402

_HIGHS = pyo.SolverFactory("appsi_highs")
needs_highs = pytest.mark.skipif(
    not _HIGHS.available(exception_flag=False), reason="HiGHS (highspy) not available"
)


def _bounded_model(lo, hi, n):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(n), bounds=(lo, hi))
    return m


def _solve(m):
    res = _HIGHS.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    return pyo.value(m.obj)


def _fit(kind, X, y):
    if kind == "LMDT":
        return LinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=15).fit(X, y)
    return HyperplaneTreeRegressor(
        max_depth=3, max_bins=6, min_samples_leaf=15, max_weight=1, num_terms=2
    ).fit(X, y)


@pytest.mark.pyomo
@needs_highs
@pytest.mark.parametrize("kind", ["LMDT", "HT"])
def test_embed_reproduces_predict(rng, kind) -> None:
    """With x fixed, the MILP's y must equal model.predict(x)."""
    X = rng.uniform(-1.5, 1.5, size=(200, 2))
    y = (X**2).sum(axis=1) + 0.3 * X[:, 0] * X[:, 1]
    model = _fit(kind, X, y)

    for p in rng.uniform(-1.4, 1.4, size=(8, 2)):
        m = _bounded_model(-1.5, 1.5, 2)
        y_out = embed_in_pyomo_model(model, m, m.x)
        for i in range(2):
            m.x[i].fix(float(p[i]))
        m.obj = pyo.Objective(expr=y_out[0])
        expected = float(np.asarray(model.predict(p[None, :]))[0])
        assert abs(_solve(m) - expected) < 1e-5, f"at {p}"


@pytest.mark.pyomo
@needs_highs
@pytest.mark.parametrize("kind", ["LMDT", "HT"])
def test_minimize_matches_dense_grid(rng, kind) -> None:
    """The MILP optimum must be at or below the best predict() on a grid."""
    X = rng.uniform(-1.5, 1.5, size=(200, 2))
    y = (X - 0.4) ** 2 @ np.array([1.0, 2.0])
    model = _fit(kind, X, y)

    m = _bounded_model(-1.5, 1.5, 2)
    y_out = embed_in_pyomo_model(model, m, m.x)
    m.obj = pyo.Objective(expr=y_out[0])
    mip_min = _solve(m)

    g = np.linspace(-1.5, 1.5, 121)
    G = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    grid_min = float(np.min(np.asarray(model.predict(G))))
    assert mip_min <= grid_min + 1e-6
    # And the reported optimum is attained: predict at x* agrees.
    x_star = np.array([[pyo.value(m.x[0]), pyo.value(m.x[1])]])
    assert abs(float(np.asarray(model.predict(x_star))[0]) - mip_min) < 1e-4


@pytest.mark.pyomo
@needs_highs
def test_multi_target_and_two_trees_in_one_model(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(200, 2))
    Y = np.stack([X[:, 0] ** 2, np.abs(X[:, 1])], axis=1)
    model = LinearTreeRegressor(max_depth=2, max_bins=6, min_samples_leaf=15).fit(X, Y)

    m = _bounded_model(-1.0, 1.0, 2)
    ya = embed_in_pyomo_model(model, m, m.x, name="a")
    yb = embed_in_pyomo_model(model, m, [m.x[1], m.x[0]], name="b")
    assert len(ya) == 2 and len(yb) == 2
    p = np.array([0.3, -0.7])
    for i in range(2):
        m.x[i].fix(float(p[i]))
    m.obj = pyo.Objective(expr=ya[0] + ya[1] + yb[0] + yb[1])
    _solve(m)
    exp_a = np.asarray(model.predict(p[None, :]))[0]
    exp_b = np.asarray(model.predict(p[None, ::-1]))[0]
    np.testing.assert_allclose([pyo.value(v) for v in ya], exp_a, atol=1e-5)
    np.testing.assert_allclose([pyo.value(v) for v in yb], exp_b, atol=1e-5)


@pytest.mark.pyomo
def test_big_m_too_small_raises(rng) -> None:
    X = rng.uniform(0.0, 10.0, size=(200, 1))
    y = np.where(X[:, 0] < 5.0, X[:, 0], 20.0 - X[:, 0])
    model = LinearTreeRegressor(max_depth=2, max_bins=8, min_samples_leaf=30).fit(X, y)
    m = _bounded_model(0.0, 10.0, 1)
    with pytest.raises(ValueError, match="too small"):
        embed_in_pyomo_model(model, m, m.x, big_m=1.0)


@pytest.mark.pyomo
def test_unbounded_inputs_warn(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = (X**2).sum(axis=1)
    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=10).fit(X, y)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(2))
    with pytest.warns(UserWarning, match="falling back"):
        embed_in_pyomo_model(model, m, m.x)

    m2 = pyo.ConcreteModel()
    m2.x = pyo.Var(range(2))
    with pytest.warns(UserWarning, match="cannot be validated"):
        embed_in_pyomo_model(model, m2, m2.x, big_m=100.0)


@pytest.mark.pyomo
def test_wrong_input_length_raises(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    model = LinearTreeRegressor(max_depth=2, min_samples_leaf=10).fit(X, X[:, 0])
    m = _bounded_model(-1.0, 1.0, 3)
    with pytest.raises(ValueError, match="expects 2"):
        embed_in_pyomo_model(model, m, m.x)
