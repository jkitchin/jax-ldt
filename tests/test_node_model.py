"""Tests for custom node models (`NodeContext`, `NodeModel`) in `grow_tree`."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt.hyperplanes import build_transform_matrix
from jax_ldt.linear_regression import add_bias
from jax_ldt.node_model import NodeContext, _check_fit_output, _check_scores
from jax_ldt.tree_core import (
    _candidate_chunk_cols,
    _evaluate_splits_chunked,
    _initial_leaf,
    grow_tree,
    reduced_split_basis,
)

# ---------------------------------------------------------------------------
# Public names and output validation
# ---------------------------------------------------------------------------


def test_names_live_in_node_model_module_only() -> None:
    # An advanced, low-level feature: importable from its module, not part of
    # the top-level namespace that everyday users see.
    import jax_ldt
    from jax_ldt.node_model import NodeContext, NodeModel  # noqa: F401

    assert "NodeContext" not in jax_ldt.__all__ and not hasattr(jax_ldt, "NodeContext")
    assert "NodeModel" not in jax_ldt.__all__ and not hasattr(jax_ldt, "NodeModel")


def test_node_context_fields() -> None:
    names = [f.name for f in dataclasses.fields(NodeContext)]
    assert names == ["X", "X_t", "y", "transform_matrix", "linear_columns", "mask", "depth", "path"]


LIN_COLS = np.array([True, True, False])


def _good_params() -> jnp.ndarray:
    return jnp.zeros((3, 2)).at[1].set(1.0)


def test_check_fit_output_accepts_valid() -> None:
    params, loss = _check_fit_output((_good_params(), jnp.asarray(0.5)), 3, 2, LIN_COLS)
    np.testing.assert_array_equal(np.asarray(params), np.asarray(_good_params()))
    assert loss == 0.5


@pytest.mark.parametrize(
    "params, loss",
    [
        (jnp.zeros((2, 2)), 0.5),  # wrong shape
        (jnp.zeros((3, 2), dtype=jnp.float32), 0.5),  # not float64
        (jnp.zeros((3, 2)).at[0, 0].set(jnp.nan), 0.5),  # not finite
        (jnp.zeros((3, 2)), jnp.inf),  # infinite loss
        (jnp.zeros((3, 2)), jnp.nan),  # NaN loss
        (jnp.zeros((3, 2)).at[2, 1].set(1.0), 0.5),  # nonzero row outside linear_columns
        (jnp.zeros((3, 2)), jnp.zeros(2)),  # loss not a scalar
    ],
    ids=["shape", "float32", "nan-param", "inf-loss", "nan-loss", "excluded-row", "loss-shape"],
)
def test_check_fit_output_rejects(params, loss) -> None:
    with pytest.raises(ValueError, match="node_model.fit"):
        _check_fit_output((params, loss), 3, 2, LIN_COLS)


@pytest.mark.parametrize("out", [_good_params(), (_good_params(),), None], ids=["array", "1-tuple", "none"])
def test_check_fit_output_requires_a_pair(out) -> None:
    with pytest.raises(ValueError, match=r"node_model.fit must return \(params, loss\)"):
        _check_fit_output(out, 3, 2, LIN_COLS)


def test_check_scores_masks_invalid() -> None:
    scores = np.zeros((2, 3))
    valid = np.ones((2, 3), dtype=bool)
    valid[0, 1] = False
    out = _check_scores(scores, 2, 3, valid)
    assert out[0, 1] == np.inf
    assert np.all(out[valid] == 0.0)
    assert np.all(scores == 0.0)  # the node model's array is not modified


def test_check_scores_accepts_negative_finite() -> None:
    out = _check_scores(np.full((2, 3), -1.0), 2, 3, np.ones((2, 3), dtype=bool))
    assert np.all(out == -1.0)


@pytest.mark.parametrize(
    "scores",
    [np.zeros((3, 2)), np.zeros((2, 3)) + np.array([0.0, np.nan, 0.0]),
     np.zeros((2, 3)) + np.array([0.0, -np.inf, 0.0])],
    ids=["transposed", "nan", "neg-inf"],
)
def test_check_scores_rejects(scores) -> None:
    with pytest.raises(ValueError, match="node_model.score_splits"):
        _check_scores(scores, 2, 3, np.ones((2, 3), dtype=bool))


# ---------------------------------------------------------------------------
# Node models in grow_tree
# ---------------------------------------------------------------------------

KW = dict(max_depth=3, max_bins=6, min_samples_leaf=10)


def _data(n_targets: int = 1, n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    Y = np.c_[np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2, X[:, 0] * X[:, 2]]
    Y = Y[:, :n_targets] + 0.05 * rng.standard_normal((n, n_targets))
    return jnp.asarray(X), jnp.asarray(Y[:, 0] if n_targets == 1 else Y)


class Replica:
    """A node model that does exactly what grow_tree does by default, given the tree's settings."""

    def __init__(self, criterion="mae", ridge=1e-5, min_samples_leaf=KW["min_samples_leaf"]):
        self.criterion, self.ridge, self.msl = criterion, ridge, min_samples_leaf

    def _X_fit(self, ctx):
        return add_bias(ctx.X_t) * jnp.asarray(ctx.linear_columns, dtype=ctx.X_t.dtype)[None, :]

    def fit(self, ctx):
        return _initial_leaf(self._X_fit(ctx), ctx.y, ctx.mask, self.ridge, self.criterion)

    def score_splits(self, ctx, thresholds, split_features, valid):
        Z, BBt, ZK = reduced_split_basis(ctx.X, ctx.transform_matrix, ctx.linear_columns)
        X_split = ctx.X_t[:, jnp.asarray(split_features, dtype=jnp.int32)]
        N, T = ctx.y.shape
        chunk = _candidate_chunk_cols(
            len(split_features), max(int(thresholds.shape[0]), 1), N, T, int(Z.shape[1])
        )
        return _evaluate_splits_chunked(
            Z, ZK, BBt, ctx.y, ctx.mask, X_split, thresholds,
            self.ridge, float(self.msl), self.criterion, chunk,
        )


class Recorder(Replica):
    """Replica that records (depth, path) per fit call; index = node id."""

    def __init__(self):
        super().__init__()
        self.fits = []
        self.scores = []

    def fit(self, ctx):
        self.fits.append((ctx.depth, ctx.path))
        return super().fit(ctx)

    def score_splits(self, ctx, thresholds, split_features, valid):
        self.scores.append((ctx.path, np.asarray(thresholds), split_features, valid))
        return super().score_splits(ctx, thresholds, split_features, valid)


def assert_trees_equal(a, b) -> None:
    for name in ("is_leaf", "feature", "threshold", "left", "right", "leaf_params",
                 "transform_matrix"):
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)),
                                      err_msg=name)
    for name in ("n", "x_mean", "x_var", "mse"):
        np.testing.assert_array_equal(np.asarray(getattr(a.leaf_uq, name)),
                                      np.asarray(getattr(b.leaf_uq, name)), err_msg=name)


def tree_paths(tree) -> dict[int, tuple]:
    """Path from the root for every node id, read from the finished tree."""
    feat, thr = np.asarray(tree.feature), np.asarray(tree.threshold)
    left, right = np.asarray(tree.left), np.asarray(tree.right)
    paths, stack = {}, [(0, ())]
    while stack:
        nid, path = stack.pop()
        paths[nid] = path
        if left[nid] >= 0:
            stack.append((int(left[nid]), path + ((int(feat[nid]), float(thr[nid]), "L"),)))
            stack.append((int(right[nid]), path + ((int(feat[nid]), float(thr[nid]), "R"),)))
    return paths


HT = dict(max_weight=1, num_terms=2)

# Each setting a node model's code depends on, covered once: growth order (path
# bookkeeping), several targets (shapes) and linear_features (the zero-row check).
CASES = {
    "defaults": dict(),
    "breadth-first-2-targets": dict(depth_first=False, n_targets=2),
    "linear-features-rmse": dict(linear_features=(0,), criterion="rmse"),
}


@pytest.mark.parametrize("hyperplane", [False, True], ids=["linear", "hyperplane"])
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_replica_reproduces_default_tree(hyperplane, case):
    kw = dict(KW, **case)
    X, y = _data(kw.pop("n_targets", 1))
    if hyperplane:
        kw["transform_matrix"] = build_transform_matrix(X, **HT)
    ours = grow_tree(X, y, node_model=Replica(kw.get("criterion", "mae")), **kw)
    assert_trees_equal(ours, grow_tree(X, y, **kw))


class OnlyFirstValid(Replica):
    def __init__(self):
        super().__init__()
        self.chosen = []

    def fit(self, ctx):
        params, _ = super().fit(ctx)
        return params, 1.0

    def score_splits(self, ctx, thresholds, split_features, valid):
        s = np.full(valid.shape, np.inf)
        k, b = np.argwhere(valid)[0]
        s[k, b] = 0.0
        self.chosen.append((split_features[k], float(np.asarray(thresholds)[b, k])))
        return s


def test_only_finite_candidate_is_chosen():
    X, y = _data()
    model = OnlyFirstValid()
    tree = grow_tree(X, y, node_model=model, **dict(KW, max_depth=1))
    assert int(tree.feature[0]) == model.chosen[0][0]
    assert float(tree.threshold[0]) == model.chosen[0][1]


class AllInf(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        return np.full(valid.shape, np.inf)


def test_all_inf_scores_make_a_leaf():
    X, y = _data()
    tree = grow_tree(X, y, node_model=AllInf(), **KW)
    assert tree.n_nodes == 1 and bool(tree.is_leaf[0])


def test_no_split_features_make_a_leaf():
    X, y = _data()
    tree = grow_tree(X, y, node_model=Replica(), split_features=(), **KW)
    assert tree.n_nodes == 1 and bool(tree.is_leaf[0])
    assert_trees_equal(tree, grow_tree(X, y, split_features=(), **KW))


class PreferInvalid(Recorder):
    def score_splits(self, ctx, thresholds, split_features, valid):
        self.scores.append((ctx.path, np.asarray(thresholds), split_features, valid))
        return np.where(valid, 0.0, -1.0)


def test_invalid_entries_are_never_chosen():
    X, y = _data()
    model = PreferInvalid()
    tree = grow_tree(X, y, node_model=model, **dict(KW, min_samples_leaf=60))
    assert any((~v).any() for _, _, _, v in model.scores), "test needs some invalid candidates"
    by_path = {p: (th, sf, v) for p, th, sf, v in model.scores}
    paths = tree_paths(tree)
    internal = np.flatnonzero(~np.asarray(tree.is_leaf))
    assert internal.size > 0
    for nid in internal:
        th, sf, v = by_path[paths[int(nid)]]
        k = sf.index(int(tree.feature[nid]))
        bs = np.flatnonzero(th[:, k] == float(tree.threshold[nid]))
        assert v[k, bs].any()


class ValidityRecorder(Replica):
    def __init__(self):
        super().__init__(min_samples_leaf=30)
        self.calls = []

    def score_splits(self, ctx, thresholds, split_features, valid):
        self.calls.append((np.asarray(ctx.X_t), np.asarray(ctx.mask), np.asarray(thresholds),
                           split_features, valid))
        return super().score_splits(ctx, thresholds, split_features, valid)


def test_validity_counts_rows_at_or_below_each_threshold():
    X, y = _data()
    X = jnp.round(X, 1)  # repeated values, so thresholds land on data values
    model = ValidityRecorder()
    grow_tree(X, y, node_model=model, **dict(KW, min_samples_leaf=30))
    assert len(model.calls) > 1
    for X_t, mask, th, sf, valid in model.calls:
        rows = X_t[mask > 0.0][:, list(sf)]
        n_below = (rows[:, None, :] <= th[None, :, :]).sum(axis=0).T
        expected = (n_below >= 30) & (len(rows) - n_below >= 30)
        assert np.isin(th, rows).any()
        np.testing.assert_array_equal(valid, expected)


@pytest.mark.parametrize("split_features", [None, (1,)])
def test_path_matches_tree(split_features):
    X, y = _data()
    model = Recorder()
    tree = grow_tree(X, y, node_model=model, split_features=split_features, **KW)
    paths = tree_paths(tree)
    assert len(model.fits) == tree.n_nodes
    for nid, (depth, path) in enumerate(model.fits):
        assert path == paths[nid]
        assert depth == len(path)
    if split_features is not None:
        assert all(sf == split_features for _, _, sf, _ in model.scores)
        assert all(f == 1 for path in paths.values() for f, _, _ in path)


class FixedLosses(Replica):
    def fit(self, ctx):
        params, _ = super().fit(ctx)
        return params, 1.0

    def score_splits(self, ctx, thresholds, split_features, valid):
        return np.full(valid.shape, 0.6)


@pytest.mark.parametrize("min_impurity_decrease, splits", [(0.39, True), (0.41, False)])
def test_min_impurity_decrease_uses_node_model_losses(min_impurity_decrease, splits):
    X, y = _data()
    tree = grow_tree(X, y, node_model=FixedLosses(), min_impurity_decrease=min_impurity_decrease,
                     **dict(KW, max_depth=1))
    assert (tree.n_nodes == 3) is splits


class JaxScores(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        return jnp.asarray(super().score_splits(ctx, thresholds, split_features, valid))


def test_scores_may_be_jax_arrays():
    X, y = _data()
    assert_trees_equal(grow_tree(X, y, node_model=JaxScores(), **KW), grow_tree(X, y, **KW))


class FailingFit(Replica):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def fit(self, ctx):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")
        return super().fit(ctx)


class FailingScore(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        raise RuntimeError("boom")


@pytest.mark.parametrize("model_cls", [FailingFit, FailingScore])
def test_node_model_exceptions_propagate(model_cls):
    X, y = _data()
    with pytest.raises(RuntimeError, match="boom"):
        grow_tree(X, y, node_model=model_cls(), **KW)


class NoScoring(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        raise AssertionError("score_splits must not be called without thresholds")


def test_no_thresholds_means_no_scoring_call():
    X, y = _data()
    tree = grow_tree(X, y, node_model=NoScoring(), **dict(KW, max_bins=1))
    assert tree.n_nodes == 1


class Float32Fit(Replica):
    def fit(self, ctx):
        params, loss = super().fit(ctx)
        return params.astype(jnp.float32), loss


class ParamsOnly(Replica):
    def fit(self, ctx):
        return super().fit(ctx)[0]


class TransposedScores(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        return super().score_splits(ctx, thresholds, split_features, valid).T


@pytest.mark.parametrize(
    "model_cls, method",
    [(Float32Fit, "node_model.fit"), (ParamsOnly, r"node_model.fit must return \(params, loss\)"),
     (TransposedScores, "node_model.score_splits")],
)
def test_malformed_node_model_output_raises(model_cls, method):
    X, y = _data()  # 3 split columns, max_bins=6 -> B=5, so (K, B) != (B, K)
    with pytest.raises(ValueError, match=method):
        grow_tree(X, y, node_model=model_cls(), **KW)


def test_hyperplane_path_uses_lifted_columns():
    X, y = _data()
    model = Recorder()
    tree = grow_tree(X, y, transform_matrix=build_transform_matrix(X, **HT), node_model=model,
                     **KW)
    paths = tree_paths(tree)
    for nid, (depth, path) in enumerate(model.fits):
        assert path == paths[nid] and depth == len(path)
    assert any(f >= X.shape[1] for path in paths.values() for f, _, _ in path)


# ---------------------------------------------------------------------------
# A node model cannot change grow_tree's own rules
# ---------------------------------------------------------------------------


class OverrideValidity(Replica):
    def score_splits(self, ctx, thresholds, split_features, valid):
        valid[...] = True
        return np.zeros(valid.shape)


class OverrideLinearColumns(Replica):
    def fit(self, ctx):
        ctx.linear_columns[...] = True
        return super().fit(ctx)


@pytest.mark.parametrize("model_cls", [OverrideValidity, OverrideLinearColumns])
def test_node_model_cannot_change_validity_or_linear_columns(model_cls):
    X, y = _data()
    with pytest.raises(ValueError, match="read-only"):
        grow_tree(X, y, node_model=model_cls(), **dict(KW, min_samples_leaf=60,
                                                       linear_features=(0,)))
