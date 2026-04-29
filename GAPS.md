# GAPS — open issues and future work

This document tracks open gaps in `jax-ldt`. The original synthesis
(2025-Q4) pulled together three independent reviews of the initial
implementation (code-quality, numerical correctness, and docs/examples
consistency). Most of the critical and high-severity items from that
pass have since landed; what remains is the medium/low/polish tail and
the deferred feature work.

The full test suite passes today. Items below are **gaps**, not
regressions — bugs, footguns, missing features, and polish that future
work should address.

Severity legend: **critical** (silent wrong answers), **high**
(user-visible failure on a documented path), **medium** (footgun /
asymmetry / partial implementation), **low** (polish), **nit**
(cosmetic).

For the list of items already resolved by prior work, see the
[Changelog](#changelog--resolved-items) at the bottom.

---

## Medium — undocumented mismatches, missing test paths, footguns

### G-19 (residual). Planned API surfaces still not delivered
The original plan's API sketch included a few surfaces that never
landed. Most planned files (viz, sklearn_compat, test_tree_core,
test_performance) are now in place; the remaining gaps are:
- `DiscoptDoE` acquisition class — promised in the plan and
  `docs/discopt_integration.md`, not implemented. (Also F-5 below.)
- Constructor-level `uq=LinearPropagationUQ()` shortcut and a
  `model.uncertainty(X)` convenience method — never landed; users must
  still call `LinearPropagationUQ().predict(model.tree_, X)`. (Also F-7.)

### G-27. `docs/algorithm.md` and `docs/active_learning.md` promise figures that don't exist
- `algorithm.md` says "with figures"; nothing is rendered.
- `active_learning.md` says "figures of the Pareto frontier of acquired
  points; comparison of acquisition strategies"; not present.
- **Fix:** add at least one matplotlib-rendered figure to each (the
  helpers in `viz.py` are now in place to back this), or relax the
  prose.

### G-28. `docs/benchmarks.md` numbers are not regenerated from the suite
- **Where:** `tests/test_performance.py` exists, but the order-of-magnitude
  table in `docs/benchmarks.md` has not been refreshed against current
  measurements.
- **Fix:** rerun the suite and update the table; consider a CI job that
  posts the numbers.

### G-36. Active learning callback shape coercion
- **Where:** `ActiveLearner.run` does `np.asarray(...).reshape(-1)` on
  the user callback's return value, which silently flattens `(N, 1)` to
  `(N,)`. Fine for single output; broken for multi-output (related to
  the multi-target AL gap below).
- **Fix:** keep the boundary check from G-9 in mind; once multi-target
  AL is supported (F-6), preserve the second axis.

---

## Low — polish / inconsistency

### G-39. Bare / broad `except` blocks
- `embed_in_discopt_model` catches `(AttributeError, TypeError)` while
  inspecting variable bounds; `from_json` catches `(OSError, ValueError)`
  while inspecting the path.
- **Fix:** narrow further or document why we swallow.

### G-40. `onnx_export.py` carries a debugging artefact
- A `Gather` op is added then immediately overwritten via
  `nodes[-1] = ...`.
- **Fix:** delete the throwaway op.

---

## Future feature work (not bugs)

### F-4. GPU benchmark notebook
A separate notebook for users with GPUs, demonstrating speed-up of the
inner JIT kernel under `jax.devices('gpu')`.

### F-5. `DiscoptDoE` acquisition class
A small class wrapping `discopt.doe.batch_optimal_experiment`, exposed
only when `discopt` is installed. Documented as advanced.

### F-6. Multi-target active learning support
The boundary check is in place (multi-target `tell` raises a clear
error), but EI/PI need to be generalised to per-target best-y plus a
scalarisation rule. Until then, the single-output limitation should
remain documented.

### F-7. `model.uncertainty(X)` convenience method
And the constructor-level `uq=LinearPropagationUQ()` shortcut from the
plan's API sketch.

### F-8. Quadric tree
Deliberately deferred. If/when a user asks for it, the feature-transform
abstraction is in place: a new `quadric_tree.py` would parallel
`hyperplane_tree.py` with quadratic lifts.

### F-9. Streaming / online refit
For large active-learning campaigns, refitting the whole tree on every
`tell` is wasteful. A warm-started growth that splits only the affected
leaves would be useful but is research.

### F-10. ONNX export via `TreeEnsembleRegressor` with a residual head
Currently the ONNX graph synthesises routing in core ops because leaves
are linear. A hybrid that uses `TreeEnsembleRegressor` for the routing
plus `MatMul` for the leaf model would shrink the graph and improve
interoperability with other ONNX-aware optimisers.

---

## Suggested triage order

1. **G-27, G-28** — docs polish (figures, refreshed benchmark table).
2. **G-39, G-40** — small polish cleanup (narrow except blocks; drop
   the throwaway ONNX Gather op).
3. **G-36** — restore multi-output shapes once F-6 lands.
4. **F-5, F-6, F-7** — planned features that real users have asked for.
5. **F-4, F-8, F-9, F-10** — long-tail / research.

Each remaining **G-** item should land with a regression test pinning
the fix; each **F-** item should land behind a feature-flag extra or
behind a clear "experimental" docstring marker.

---

## Changelog — resolved items

The following items from earlier review passes have been resolved by
intervening work and are kept here for historical reference.

**Critical (saved-model correctness, persistence, statistical validity):**
- **G-1.** Saved JSON `kind` field — `to_json` now infers
  `linear_tree` / `hyperplane_tree` from the regressor class via
  `_infer_kind` in `src/jax_ldt/export/spec.py`.
- **G-2.** PRNG round-trip on `ActiveLearner.save → load` — loop now
  persists `seed` + `rng_step` and reconstructs keys with
  `jax.random.fold_in(PRNGKey(seed), step)`; v1 payloads still load via
  a fallback.
- **G-3.** Mondrian conformal sparse-leaf strategy — `conformal.py`
  now exposes `sparse_leaf_strategy: {"global", "skip"}` with the
  exchangeability tradeoff documented.

**High (public-facing bugs on documented paths):**
- **G-4.** `msle` and `max_abs` criteria — implemented as
  `_weighted_msle` / `_weighted_max_abs` in `tree_core.py`.
- **G-5.** `min_impurity_decrease` semantics — comparison rewritten
  to `(node_loss - best_loss) < min_impurity_decrease`; matches sklearn.
- **G-6.** Asymmetric UQ-class API — `LinearPropagationUQ.predict`,
  `QuadraticUQ.calibrate/predict`, and `ConformalCalibrator.calibrate`
  all accept `model_or_tree`.
- **G-7.** Empty-data fit — `_validation.py` raises `ValueError` for
  empty `X`, length mismatch, and bad shapes.
- **G-8.** JIT compile invariants — `_fit_leaf` is now an alias for
  `_fit_one_side`, eliminating the redundant trace.
- **G-9.** `ActiveLearner` multi-output handling — `tell` raises a
  clear error on multi-target inputs.
- **G-10.** Notebook 02 coverage claim — markdown caveats the
  marginal / finite-sample nature of the guarantee and quotes the
  `±√(α(1-α)/N_test)` band; calibration / test sizes were bumped
  (N=4000, ~800 test) so executed runs land in-band (current run prints
  0.932 against a 0.90 target).
- **G-38.** Notebook 01 HT vs LMDT narrative — rewritten to
  "comparable at this leaf budget" with a pointer to a diagonal target
  (and notebook 06) where HT wins decisively.

**Medium / numerical hardening (this pass):**
- **G-14.** ONNX round-trip tolerance — `tests/test_export_onnx.py`
  documents the empirical `atol=1e-8` choice in a comment and adds a
  depth-8 stress test on Friedman-1 data with `atol=1e-7`.
- **G-15.** Float32 inputs bypassing X64 — `tree_core.grow_tree`,
  `predict`, and `apply_tree` now cast `X`/`y` to `jnp.float64` at
  the boundary; the regressor wrappers continue to warn on non-float64
  inputs, then upcast. Float32 inputs no longer silently downgrade
  precision.
- **G-20.** Regressor docstrings — `LinearTreeRegressor` /
  `HyperplaneTreeRegressor` already carry per-parameter one-liners; the
  remaining "mirror upstream" placeholder in `tree_core.grow_tree` was
  replaced with a Numpy-style block that points users at the regressor
  classes for the public surface.
- **G-22.** Untested public surface — `tests/test_acquisitions.py`,
  `tests/test_categorical_features.py`, and the new
  `tests/test_smoke_coverage.py` cover `ProbabilityOfImprovement`,
  `MaxVariance`, `categorical_features` at the regressor level,
  `min_samples_leaf` as a float fraction, and HT in
  `embed_in_discopt_model`.
- **G-24.** `mypy --strict` declared but the public API isn't fully
  annotated — the `# type: ignore[arg-type]` suppressions in
  `_types.py` are gone (the pytree wire types are now typed tuples);
  `ArrayLike` on the regressors is properly parameterised. Wider
  `--strict` cleanup across the rest of the package is its own task.
- **G-31.** Naming inconsistencies — regressors now expose `n_leaves`
  as an alias for `num_leaves` so it matches `Tree.n_leaves`;
  `tree_core.apply_` is already aliased to `apply_tree`.
- **G-35.** `[dev]` extra split — `pyproject.toml` `[dev]` is now
  lint+test only; a `[dev-full]` meta-extra recreates the previous
  bundle, and consumers compose `[dev,onnx,viz,sklearn]` /
  `[dev,parity]` for richer subsets.
- **G-41.** Verified-working items still without regression tests —
  `tests/test_smoke_coverage.py` adds checks that ONNX export passes
  `onnx.checker.check_model` and that `predict`/`apply`/`num_leaves`
  raise before `fit`. `calibration_metrics`,
  `ConformalCalibrator(alpha=invalid)`, and
  `to_discopt_decision_tree` rejecting linear leaves were already
  covered by existing tests.

**Medium / numerical hardening:**
- **G-11.** Linear-propagation σ at `n=1` and constant features —
  `linprop.py` now drops the deviation term when `n_eff < 2` and skips
  constant-variance features from the deviation sum.
- **G-12.** Hyperplane dedup loss — `hyperplanes.py` warns when
  rounding-then-dedup drops more than 10% of rows.
- **G-13.** discopt big-M derivation — already marked resolved in the
  prior pass; bounds are derived from `x_vars.lb/ub` per node and per
  leaf, with a `UserWarning` on fallback.
- **G-16.** NaN / Inf input validation — `_validation._check_finite`
  is called at every public `fit` / `predict` / `calibrate` boundary.
- **G-17.** HT/LMDT parameter symmetry — `HyperplaneTreeRegressor`
  now exposes `linear_features` and `split_features`, threaded through
  to `grow_tree`.
- **G-18.** `MaxVariance` — has a real `score` method (no longer an
  empty dataclass), even if the implementation is short.
- **G-19 (planned files).** `viz.py`, `sklearn_compat.py`,
  `tests/test_tree_core.py`, and `tests/test_performance.py` all
  landed; the remaining residual is captured above.
- **G-21.** `to_json` `kind` parameter — now documented in the
  function docstring.
- **G-23.** Tracked `egg-info` directory — removed; `*.egg-info/` is
  in `.gitignore`.
- **G-25.** Unused build dependencies — `tqdm` and `setuptools_scm`
  removed from `pyproject.toml`.
- **G-26.** `docs/active_learning.md` sugar example — now shows the
  prerequisite `loop.tell(X_init, y_init)` priming step.
- **G-29.** `predict_tree` re-export — exposed at the top-level
  `jax_ldt.__init__` and listed in `__all__`.
- **G-30.** Hyperparameter inconsistency between SKILL.md and
  quickstart.md — both now use `max_weight=1`.
- **G-32.** Sparse-leaf warning truncation — message now includes
  `(+N more)` when the truncation kicks in.
- **G-33.** `hyperplane-decision-trees.pdf` — moved to
  `docs/references/`.
- **G-34.** Empty `_dev/` directory — removed (also marked resolved
  in the prior pass).
- **G-37.** Constant-target σ in linprop — addressed alongside G-11;
  σ is 0 when `mse == 0` and `n_eff ≥ 2`.

**Features delivered:**
- **F-1.** `sklearn_compat` module — landed.
- **F-2.** `viz.py` plotting helpers — landed.
- **F-3.** `tests/test_performance.py` benchmark suite — landed
  (refresh of `docs/benchmarks.md` table is tracked under G-28).
