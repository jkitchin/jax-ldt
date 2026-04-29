"""Coverage for items pinned in GAPS.md G-22 / G-41.

These exports were spot-checked in review and found correct, but lacked
dedicated regression tests. The tests here are deliberately narrow
smoke tests — their job is to fail loudly if the surface goes away or
flips behavior, not to re-derive the underlying correctness proofs.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor


@pytest.mark.parametrize("regressor", [LinearTreeRegressor, HyperplaneTreeRegressor])
def test_methods_raise_before_fit(regressor) -> None:
    """`predict`, `apply`, and `num_leaves` must error before `fit`."""
    model = regressor()
    X = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match=r"fit"):
        model.predict(X)
    with pytest.raises(RuntimeError, match=r"fit"):
        model.apply(X)
    with pytest.raises(RuntimeError, match=r"fit"):
        _ = model.num_leaves


def test_n_leaves_alias_matches_num_leaves(branin_2d) -> None:
    """G-31: regressors expose both spellings; they must agree."""
    X, y = branin_2d
    lmdt = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    ht = HyperplaneTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=20, max_weight=1, num_terms=2
    ).fit(X, y)
    assert lmdt.n_leaves == lmdt.num_leaves
    assert ht.n_leaves == ht.num_leaves


@pytest.mark.onnx
def test_onnx_export_passes_checker(branin_2d) -> None:
    """G-41: the synthesised graph must pass `onnx.checker.check_model`."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from jax_ldt import to_onnx

    X, y = branin_2d
    model = LinearTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=20
    ).fit(X, y)
    proto = to_onnx(model)
    onnx.checker.check_model(proto)


@pytest.mark.discopt
def test_embed_ht_in_discopt_model(rng) -> None:
    """G-22: HT was untested in the discopt adapter; LMDT had a test.

    We don't probe optimality here (HT's lifted features make the
    piecewise minimum non-trivial to verify with scipy); we just check
    that the embed builds and solves to a finite objective inside the
    in-sample range.
    """
    discopt = pytest.importorskip("discopt")
    from jax_ldt.export import embed_in_discopt_model

    X = rng.uniform(-1.5, 1.5, size=(160, 2))
    y = (X ** 2).sum(axis=1) + 0.05 * X[:, 0] * X[:, 1]
    model = HyperplaneTreeRegressor(
        max_depth=3, max_bins=4, min_samples_leaf=20, max_weight=1, num_terms=2
    ).fit(X, y)

    m = discopt.Model("ht-embed")
    x = m.continuous("x", shape=(2,), lb=-1.5, ub=1.5)
    y_expr = embed_in_discopt_model(model, m, x, big_m=20.0)
    m.minimize(y_expr[0])
    result = m.solve()
    obj = float(result.objective)
    # Surrogate is non-negative on this bowl; finite, in-range solution.
    assert np.isfinite(obj)
    assert obj <= float(np.max(y)) + 1e-3
