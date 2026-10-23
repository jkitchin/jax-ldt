---
name: jax-ldt
description: Pure-JAX linear and hyperplane decision trees (port of LLNL's hyperplanetree). Piecewise-linear surrogate modeling with first-class uncertainty quantification (linear-propagation, quadratic-difference, Mondrian leaf conformal), batched active learning (ask/tell + run sugar with campaign persistence), and MIP-friendly export (solver-neutral JSON, hand-built ONNX, optional discopt adapter for direct MINLP embedding without Pyomo).
allowed-tools: ["*"]
---

# jax-ldt — pure-JAX (Hyperplane) Decision Trees

## Overview

`jax-ldt` is a JAX rewrite of LLNL's PyTorch
[`hyperplanetree`](https://github.com/LLNL/systems2atoms/tree/main/systems2atoms/hyperplanetree).
It builds **piecewise-linear regression surrogates**: decision trees
where every leaf holds a linear model, and (in the hyperplane variant)
splits can follow oblique directions, not just axis-aligned cuts.

Key differences from upstream:

- Pure JAX numerics — `jnp` arrays end-to-end, JIT on the hot inner
  split kernel, vmapped ridge fits, `float64` throughout (requires
  `JAX_ENABLE_X64=1`, enforced at fit time).
- **No Pyomo / OMLT runtime dependency.** Trained models export to
  JSON or ONNX (consumable by OMLT externally) and embed directly in
  `discopt` MINLPs.
- First-class **uncertainty quantification** with three composable
  methods (linear propagation, quadratic difference, Mondrian leaf
  conformal — distribution-free with valid coverage).
- **Batch active learning loop** with `ask`/`tell` primitives,
  campaign-file persistence, and a `run` callback sugar.

## When to Use

**Use `jax-ldt` when:**
- You need a fast, MIP-embeddable regression surrogate (single
  features → single output, or multi-output).
- Piecewise-linear behaviour is acceptable or desirable (you'll
  later embed it in a MILP / MINLP).
- You want honest uncertainty intervals with valid coverage
  (Mondrian leaf conformal).
- You have a slow / expensive experiment and want batched active
  learning to direct the next round.
- You want to optimise *over* a learned surrogate (e.g., minimise
  predicted cost subject to constraints) using `discopt`.

**Don't use it when:**
- You want a smooth surrogate (use a Gaussian process or neural net).
- The function has discontinuities / categorical structure that fits
  poorly into linear leaves (try a regular DT or NN).
- You need ensembling out of the box (we ship single trees only).

## Installation

Not published on PyPI — install from the repository:

```bash
pip install "jax-ldt @ git+https://github.com/jkitchin/jax-ldt.git"
# or, from a checkout:
pip install -e ".[dev,onnx]"
```

64-bit JAX must be enabled **before** `jax` / `jax_ldt` is imported
(`os.environ["JAX_ENABLE_X64"] = "1"`); otherwise every fit raises a
`RuntimeError`.

Optional extras:

| extra | adds | use |
|-------|------|-----|
| `[onnx]` | `onnx`, `onnxruntime` | export + round-trip |
| `[discopt]` | `discopt>=0.3` | MINLP embedding |
| `[viz]` | `matplotlib` | plot helpers |
| `[uq-metrics]` | `uncertainty-toolbox` | calibration / sharpness metrics |
| `[dev]` | `pytest`, `torch`, `scikit-learn`, … | parity tests + tooling |

## Public API at a Glance

```python
from jax_ldt import (
    LinearTreeRegressor,            # axis-aligned LMDT
    HyperplaneTreeRegressor,        # oblique-split HT
    # uncertainty
    LinearPropagationUQ,
    QuadraticUQ,
    ConformalCalibrator,
    fit_with_conformal,
    # active learning
    ActiveLearner,
    ExpectedImprovement,
    UncertaintySampler,
    GreedyMaxMinBatchSelector,
    DiverseBatchSelector,
    # export
    to_json, from_json, to_onnx,
)
from jax_ldt.export import (
    embed_in_discopt_model,         # full MINLP embedding
    to_discopt_decision_tree,       # constant-leaf-only
)
```

## Core Patterns

### 1. Fit and predict

```python
import jax.numpy as jnp
from jax_ldt import HyperplaneTreeRegressor

model = HyperplaneTreeRegressor(
    max_depth=8, max_bins=10, min_samples_leaf=15,
    max_weight=1, num_terms=2, ridge=1e-5,
)
model.fit(X, y)               # X, y: ndarray or jnp.ndarray
yh = model.predict(X_new)     # JAX-jit-compiled inner kernel
sigma = LinearPropagationUQ().predict(model.tree_, X_new)
```

`HyperplaneTreeRegressor` is the right default for ≥2D problems.
`LinearTreeRegressor` skips the hyperplane lift; use it for 1D inputs
or when you specifically want axis-aligned cuts.

### 2. Conformal prediction intervals

The **two-step** API: fit on training, calibrate on a held-out set.

```python
from jax_ldt import HyperplaneTreeRegressor, ConformalCalibrator

model = HyperplaneTreeRegressor(...).fit(X_train, y_train)
calib = ConformalCalibrator(
    alpha=0.1,                       # 90% intervals
    mondrian=True,                   # per-leaf quantile
    min_calibration_per_leaf=5,      # sparse-leaf fallback threshold
).calibrate(model, X_cal, y_cal)
lo, hi = calib.predict_interval(X_test, model=model)
```

Convenience: `fit_with_conformal(model, X, y, calibration_size=0.2)`
splits internally and returns `(model, calib)`.

**Sparse-leaf fallback**: leaves with fewer than
`min_calibration_per_leaf` calibration points fall back to the global
quantile and emit a single warning naming them.

### 3. Batch active learning

`ask` / `tell` is the primitive; `run` is sugar.

```python
import numpy as np
from jax_ldt import (
    ActiveLearner, HyperplaneTreeRegressor,
    ExpectedImprovement, GreedyMaxMinBatchSelector,
)

bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)
loop = ActiveLearner(
    model_factory=lambda: HyperplaneTreeRegressor(max_depth=5),
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    bounds=bounds,
    batch_size=8,
    seed=42,
)

# initial data
loop.tell(X0, y0)

# manual: ask, run lab, tell back
batch_id, X_batch = loop.ask()
loop.save("campaign.json")        # survives a crashed session
y_partial = run_lab(X_batch[[0, 2, 5]])
loop.tell(batch_id, y_partial, indices=[0, 2, 5])  # PARTIAL responses OK

# or: in-silico sugar
loop.run(run_experiment=lambda X: f(X), n_rounds=10)
```

**Campaign file** persists data, RNG, history, and pending batches —
**not** the trained model (it's refit on `load`). This decouples the
saved campaign from internal `Tree` layout.

**Index-aligned tell**: `tell(batch_id, y, indices=[…])` accepts a
subset of the original batch. Unanswered rows stay pending; the next
`ask()` may re-propose or replace them. No silent data drops.

### 4. Acquisitions and batchers

| acquisition | use when | multi-target |
|-------------|----------|--------------|
| `UncertaintySampler` | pure exploration; score = σ | yes |
| `MaxVariance` | pure exploration; score = σ² (argmax-equivalent) | yes |
| `ExpectedImprovement(direction="min")` | classic Bayesian optimisation | no |
| `ProbabilityOfImprovement` | conservative variant of EI | no |

**Multi-target campaigns.** `tell` accepts `y` of shape `(n, k)`. The
single-target constraint belongs to the acquisition, not the data: each
acquisition declares `requires_scalar_target`, and only a scalar-only
acquisition (EI / PI, which need a scalar best-y) paired with `(n, k)`
data raises. Per-target σ is reduced to one score per candidate by
dividing by `target_scales` — ideally the tolerance required on each
target — and taking the per-candidate `max` (or `mean` via
`reduce="mean"`). Left as `None`, the loop substitutes each target's
observed standard deviation each round.

```python
UncertaintySampler(target_scales=[0.01, 5.0])   # tight on target 0
```

| batcher | strategy |
|---------|----------|
| `TopKBatchSelector` | naive top-k by score |
| `GreedyMaxMinBatchSelector` | score weighted by min-distance to chosen — **recommended default** |
| `DiverseBatchSelector` | KMeans-cluster top-N candidates, pick highest-scoring per cluster |

### 5. Export and MIP embedding

```python
from jax_ldt import to_json, from_json, to_onnx

to_json(model, "tree.json")          # canonical, human-readable
to_onnx(model, "tree.onnx")          # for onnxruntime / OMLT (external)
```

For **direct MINLP optimisation** without Pyomo, embed in `discopt`:

```python
import discopt
from jax_ldt.export import embed_in_discopt_model

m = discopt.Model("min-y")
x = m.continuous("x", shape=(n_features,), lb=lb, ub=ub)
y_expr = embed_in_discopt_model(model, m, x, big_m=20.0)
m.minimize(y_expr[0])
result = m.solve()
print("argmin:", result.x["x"], "min y:", result.objective)
```

The adapter encodes:
1. lifted-feature equality constraints `x'_j = Σ A[i,j] x_i`,
2. one binary `z_k` per leaf with `Σ z_k = 1`,
3. big-M routing constraints for every internal split,
4. bilinear leaf regressions `v_k = z_k · (β_k · x' + α_k)`
   linearised with big-M.

This is the standard "GDP via big-M" pattern OMLT uses, but built
directly with discopt primitives.

## Hyperparameter Cheat Sheet

| param | default | effect |
|-------|---------|--------|
| `max_depth` | 32 | tree depth cap |
| `max_bins` | 10 (HT) / 25 (LMDT) | quantile cuts per feature |
| `min_samples_leaf` | 0.01 (frac) | min rows per leaf |
| `min_samples_split` | 6 | min rows to attempt split |
| `ridge` | 1e-5 | ridge regularisation in leaf fits |
| `criterion` | "mae" | one of mae / mse / rmse / msle / max_abs |
| `max_weight` (HT) | 1 | int weight cap in hyperplane enumeration |
| `num_terms` (HT) | 2 | features combined per oblique split |

Practical rules of thumb:

- For 6+ features, keep `num_terms=2` and `max_weight ∈ {1, 2}`. Higher
  values explode the lifted-feature count combinatorially. Check the
  count before committing to a fit — `count_lifted_features(n_features,
  num_terms=..., max_weight=...)` returns it without fitting, and fit
  time tracks it (≤1000 → seconds; >5000 → minutes to hours). Cap it
  with `max_hyperplanes=...`. See `docs/benchmarks.md` ("Cost model").
- `min_samples_leaf=15` is a sane default for noisy real data; smaller
  for clean simulators, larger for human-collected lab data.
- `max_depth=5–8` covers most usable regimes; deeper trees overfit
  before they help (per the upstream paper).

## Architecture / Internals

- **Tree pytree** (`jax_ldt._types.Tree`): plain `@dataclass(frozen=True)`
  registered with `jax.tree_util.register_pytree_node`. Parallel-array
  layout: `is_leaf`, `feature`, `threshold`, `left`, `right`,
  `leaf_params`, `transform_matrix`, `leaf_uq`. Matches discopt's
  `DecisionTree` layout for cheap conversion.
- **Tree growth** (`tree_core.grow_tree`): Python while-loop over a
  queue of node ids (data-dependent topology — not JIT-compatible).
  Each step calls a JIT-compiled inner kernel.
- **Inner split kernel** (`_evaluate_splits_kernel`): vmapped over
  `K * B` candidate splits, each a ridge fit on a side-mask. Shape-
  stable so JIT compiles once. Uses mask multiplication over the full
  sample axis, not slicing. Each candidate is solved in the
  `m = n_features_in + 1` dimensional space the lifted design matrix
  actually spans — `X_fit = Z @ B` with `Z = [1, X]`, so the
  push-through identity `(BᵀGB + λI)⁻¹Bᵀv = Bᵀ(G·BBᵀ + λI)⁻¹v` applies.
  Exact, and independent of the lift width, which is what keeps
  hyperplane fits from blowing up (issue #2). Candidates are evaluated
  in column chunks under a fixed memory budget
  (`tree_core.SPLIT_MEMORY_BUDGET_BYTES`); ordinary problems take one
  chunk and are bit-identical to an unchunked evaluation.
- **Node fits** (`_add_node` / `_finalize_leaf`): a node's regression is
  solved once, at creation, and reused as both its impurity baseline and
  its leaf model. Leaf parameters are solved densely in the lifted space
  (`_fit_one_side`, eager) — those coefficients are the exported model,
  so reproducibility outranks the cubic cost there.
- **Hyperplane lifting** (`hyperplanes.build_transform_matrix`):
  enumerates integer-weight rows via Miller indices, symmetrises (parity
  + permutation), de-duplicates at `tol_decimals` precision, prepends
  identity, optionally rescales by feature ranges. Result: matrix `A`
  such that `X' = X @ A` is the lifted feature space. `max_hyperplanes`
  caps the oblique block (evenly spaced subsequence of the lex-sorted
  rows); a `UserWarning` fires past
  `LIFTED_FEATURE_WARN_THRESHOLD` so an intractable configuration
  reports itself instead of hanging.

## Common Pitfalls

- **`num_terms > n_features`**: hyperplane enumeration raises if
  `num_terms` exceeds the number of non-categorical features. For 1D,
  use `LinearTreeRegressor`.
- **Bumping `num_terms` "just to try it"**: it multiplies the lifted
  feature count, and the per-node leaf solve is cubic in that count. Run
  `count_lifted_features(...)` first — going from `num_terms=2` to `4`
  on 8 features takes the lift from 64 columns to 10144.
- **JIT recompile cost on first call**: ~0.5 s on CPU for the inner
  kernel. Re-fitting trees with the same hyperparameters reuses the
  cached compilation.
- **Conformal coverage on tiny calibration sets**: with < 20 calibration
  points, intervals can be wider than expected because the empirical
  quantile correction needs `ceil((N+1)(1-α))/N`. For Mondrian, prefer
  ≥ 5 points per leaf and tune `min_calibration_per_leaf`.
- **discopt embedding and big-M**: leave `big_m=None` to compute a
  bound from the variable box. If the user's `discopt.continuous(...)`
  is unbounded, supply an explicit `big_m=...` to keep the solver happy.

## Examples (executed notebooks)

The `examples/` directory contains five end-to-end notebooks executed
in place — open them to see plots and outputs:

| notebook | covers |
|---|---|
| `01_quickstart.ipynb` | fit LMDT and HT on Branin, side-by-side surface plot |
| `02_uncertainty.ipynb` | linprop, quadratic, Mondrian conformal with 90% coverage check |
| `03_active_learning.ipynb` | EI loop on a 2D quadratic, save/resume campaign |
| `04_mip_optimization.ipynb` | discopt MINLP that minimises a learned tree |
| `05_onnx_export.ipynb` | round-trip via onnxruntime to 1e-9 |
| `06_steam_tables.ipynb` | active-learning loop trains a PHT (P, h → T) surrogate against IAPWS-97; reproduces the paper's Figure 9 phase-diagram comparison |

The steam-table notebook needs the optional `[steam]` extra
(`pip install -e ".[steam,onnx]"`) which adds the `iapws` package.

To regenerate after editing source: `python examples/_make_notebooks.py`
followed by `jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb`.

## Tests

```bash
pytest tests/                         # unit + parity + property (~1 min)
pytest tests/test_parity_vs_pytorch.py # equivalence vs upstream PyTorch
pytest tests/test_principled_choices.py # design-rule pin tests
```

The parity test asserts **predictive equivalence** (not topology
equality): train MAE within ±50% of upstream and held-out grid Pearson
r > 0.95. Greedy tree growth diverges with even tiny numerical
perturbations, so topology-pinning would be brittle.

## Related Projects

- **Upstream PyTorch**: <https://github.com/LLNL/systems2atoms/tree/main/systems2atoms/hyperplanetree>.
  Original source for LMDT + HT + OMLT integration.
- **OMLT**: <https://github.com/cog-imperial/OMLT>. Pyomo-based MINLP
  embedding for ML surrogates. Can consume our `to_onnx` output
  externally.
- **discopt**: pure-JAX MINLP solver. We provide a direct adapter
  (`embed_in_discopt_model`) so you can solve over a learned tree
  without leaving Python.
- **Paper**: Sunshine et al., *Hyperplane decision trees as piecewise
  linear surrogate models for chemical process design*, Comput. Chem.
  Eng. 202 (2025) 109204.

## Quick Sanity Test

```python
import numpy as np
from jax_ldt import HyperplaneTreeRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(200, 2))
y = X[:, 0] + X[:, 1] + 0.05 * rng.standard_normal(200)
model = HyperplaneTreeRegressor(max_depth=4, max_bins=6).fit(X, y)
assert float(np.mean(np.abs(model.predict(X) - y))) < 0.2
```

If this passes, your install is good.
