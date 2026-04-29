"""JSON serialization round-trip tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor, from_json, to_json
from jax_ldt.export.spec import tree_from_dict, tree_to_dict
from jax_ldt.tree_core import predict


def test_lmdt_json_roundtrip(branin_2d, tmp_path) -> None:
    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=15).fit(X, y)
    s = to_json(model, tmp_path / "tree.json")
    assert (tmp_path / "tree.json").exists()
    tree = from_json(tmp_path / "tree.json")
    yh_orig = np.asarray(model.predict(X))
    yh_round = np.asarray(predict(tree, X))
    np.testing.assert_allclose(yh_orig, yh_round, atol=1e-12)


def test_ht_json_roundtrip(branin_2d) -> None:
    X, y = branin_2d
    model = HyperplaneTreeRegressor(
        max_depth=4, max_bins=6, min_samples_leaf=15, max_weight=1, num_terms=2
    ).fit(X, y)
    s = to_json(model)
    tree = from_json(s)
    yh_orig = np.asarray(model.predict(X))
    yh_round = np.asarray(predict(tree, X))
    np.testing.assert_allclose(yh_orig, yh_round, atol=1e-12)


def test_dict_form_is_json_serializable(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    d = tree_to_dict(model.tree_)
    # Should round-trip through json without error.
    s = json.dumps(d)
    d2 = json.loads(s)
    tree = tree_from_dict(d2)
    yh1 = np.asarray(model.predict(X))
    yh2 = np.asarray(predict(tree, X))
    np.testing.assert_allclose(yh1, yh2, atol=1e-12)


def test_json_schema_keys(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=2, max_bins=3, min_samples_leaf=10).fit(X, y)
    d = tree_to_dict(model.tree_)
    assert d["version"] == 1
    assert "transform_matrix" in d
    assert "tree" in d
    assert set(d["tree"].keys()) == {
        "node_count", "is_leaf", "feature", "threshold", "left", "right", "leaf_params"
    }
    # NaN in threshold becomes null in JSON.
    s = to_json(model)
    assert "null" in s


def test_kind_field_inferred_from_regressor(toy_1d, branin_2d) -> None:
    X1, y1 = toy_1d
    X2, y2 = branin_2d
    lmdt = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X1, y1)
    ht = HyperplaneTreeRegressor(
        max_depth=3, max_bins=5, min_samples_leaf=15, max_weight=1, num_terms=2
    ).fit(X2, y2)

    s_lmdt = to_json(lmdt)
    s_ht = to_json(ht)

    assert '"kind": "linear_tree"' in s_lmdt
    assert '"kind": "hyperplane_tree"' in s_ht


def test_kind_explicit_override(toy_1d) -> None:
    X, y = toy_1d
    model = LinearTreeRegressor(max_depth=2, max_bins=3, min_samples_leaf=10).fit(X, y)
    s = to_json(model, kind="my_custom_label")
    assert '"kind": "my_custom_label"' in s
