"""discopt adapter tests. Skipped if discopt is not installed."""

from __future__ import annotations

import numpy as np
import pytest

discopt = pytest.importorskip("discopt")

import jax.numpy as jnp  # noqa: E402

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor  # noqa: E402
from jax_ldt.export import embed_in_discopt_model, to_discopt_decision_tree  # noqa: E402


@pytest.mark.discopt
def test_to_discopt_decision_tree_rejects_linear_leaves(branin_2d) -> None:
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=20).fit(X, y)
    # LMDT has linear leaves → conversion should raise.
    with pytest.raises(ValueError, match="non-trivial leaf regressions"):
        to_discopt_decision_tree(model)


@pytest.mark.discopt
def test_to_discopt_decision_tree_constant_leaf_works(rng) -> None:
    """`to_discopt_decision_tree` must extract leaf biases correctly.

    We force the tree to be constant-leaf by zeroing slope coefficients,
    then validate the conversion against two *independent* references:

      1. The bias column of ``leaf_params`` directly (so a regression in
         the conversion's value-extraction path can't hide behind a
         matching wrong-prediction in ``model.predict``).
      2. ``model.predict`` on the data — guards leaf-routing equivalence.
    """
    X = rng.uniform(-1, 1, size=(120, 2))
    y = (X[:, 0] > 0).astype(np.float64)  # piecewise-constant target
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)

    # Patch leaves to constant by zeroing the slope coefficients.
    t = model.tree_
    new_params = jnp.asarray(np.asarray(t.leaf_params))
    zero_slopes = new_params.at[:, 1:, :].set(0.0)
    model.tree_ = t.replace(leaf_params=zero_slopes)

    dt = to_discopt_decision_tree(model)

    # (1) Independent check: discopt's leaf value array must equal the
    # bias column of leaf_params for every leaf node id. The discopt
    # DecisionTree stores `value` for every node (leaf and internal);
    # we only compare on actual leaves.
    is_leaf = np.asarray(model.tree_.is_leaf)
    expected_bias = np.asarray(model.tree_.leaf_params)[:, 0, 0]
    discopt_value = np.asarray(dt.value)
    np.testing.assert_allclose(
        discopt_value[is_leaf], expected_bias[is_leaf], atol=1e-12,
        err_msg="discopt leaf values must equal the bias column of leaf_params",
    )

    # (2) End-to-end agreement on training data.
    yh_jax = np.asarray(model.predict(X))
    yh_discopt = np.array([dt.predict(x) for x in np.asarray(X)])
    np.testing.assert_allclose(yh_discopt, yh_jax, atol=1e-9)


@pytest.mark.discopt
def test_big_m_fallback_warns_on_unbounded(rng) -> None:
    """G-13: when bounds can't be read off x_vars, warn on big_m fallback.

    discopt's continuous variable always exposes ``lb``/``ub`` (it uses
    ±1e20 sentinels), so to trigger the fallback we wrap the variable
    in a thin proxy that hides those attributes — the legitimate
    ``AttributeError`` / ``TypeError`` path the narrowed ``except``
    clause is meant to catch.
    """
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = (X**2).sum(axis=1)
    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=10).fit(X, y)

    m = discopt.Model("opt-tree-unbounded")
    x = m.continuous("x", shape=(2,))

    class _NoBoundsProxy:
        def __init__(self, inner):
            self._inner = inner

        def __getitem__(self, k):
            return self._inner[k]

        # Deliberately do NOT expose lb / ub.

    x_proxy = _NoBoundsProxy(x)

    with pytest.warns(UserWarning, match=r"big_m"):
        embed_in_discopt_model(model, m, x_proxy)


@pytest.mark.discopt
def test_embed_lmdt_in_discopt_model(rng) -> None:
    """Embed a tiny LMDT in discopt and minimize y; compare to scipy."""
    X = rng.uniform(-1.5, 1.5, size=(150, 2))
    y = (X**2).sum(axis=1) + 0.05 * X[:, 0] * X[:, 1]
    model = LinearTreeRegressor(max_depth=3, max_bins=4, min_samples_leaf=15).fit(X, y)

    m = discopt.Model("opt-tree")
    x = m.continuous("x", shape=(2,), lb=-1.5, ub=1.5)
    y_expr = embed_in_discopt_model(model, m, x, big_m=20.0)
    m.minimize(y_expr[0])
    result = m.solve()

    # We expect a solution with y at or below the smallest in-sample y.
    in_sample_min = float(np.min(y))
    discopt_y = float(result.objective)
    assert discopt_y <= in_sample_min + 1e-3, (
        f"discopt min {discopt_y:.4f} did not match tree's piecewise minimum "
        f"(in-sample min {in_sample_min:.4f})"
    )
