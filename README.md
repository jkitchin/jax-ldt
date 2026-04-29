# jax-ldt

![jax-ldt logo](jax-ldt.png)

[![CI](https://github.com/jkitchin/jax-ldt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/jax-ldt/actions/workflows/ci.yml)



Pure-JAX linear and hyperplane decision trees for piecewise-linear
surrogate modeling, with first-class uncertainty quantification and a
batched active-learning loop.

This is a JAX rewrite of the PyTorch
[hyperplanetree](https://github.com/LLNL/systems2atoms/tree/main/systems2atoms/hyperplanetree)
library, omitting the Pyomo/OMLT runtime dependency. Trained models
export to a solver-neutral JSON spec and to ONNX, which can still be
consumed by OMLT externally for MILP embedding.

This implementation was motivated by this work: Ethan M. Sunshine, Carolina Colombo Tedesco, Sneha A. Akhade, Matthew J. McNenly, John R. Kitchin, and Carl D. Laird, Hyperplane Decision Trees as Piecewise Linear Surrogate Models for Chemical Process Design, Computers & Chemical Engineering, (2025). https://doi.org/10.1016/j.compchemeng.2025.109204. 

## What is a hyperplane tree?

A **linear-model decision tree (LMDT)** is a decision tree where every
leaf holds a linear regression rather than a constant. The tree
partitions the input space, and within each region a small linear
model approximates the target — so the global surrogate is
**piecewise linear**.

A **hyperplane tree (HT)** generalises this by letting splits run
along **oblique directions** instead of being restricted to one
feature at a time. Internally we lift the input through a
transformation matrix `A` (built from integer-weight combinations up
to `max_weight` over `num_terms` features) and grow an axis-aligned
tree on `X' = X @ A`. Each internal split is therefore a hyperplane
`A[:, f]·x = t` in the original feature space — the diagonal cuts you
see in the figure below.

Why use one?

- Trains in seconds on small/medium tabular data, with a JIT-compiled
  inner split kernel that vmaps a ridge fit over every candidate.
- The piecewise-linear structure embeds **exactly** in MILP / MINLP
  via big-M routing — no over-approximation, no Lipschitz smoothing.
- Honest uncertainty intervals are cheap: every leaf already owns a
  linear model and a calibration set of residuals.

## Install

```bash
pip install -e ".[dev,onnx]"
```

Optional extras:

| extra | adds | use |
|-------|------|-----|
| `[onnx]` | `onnx`, `onnxruntime` | export + round-trip |
| `[discopt]` | `discopt>=0.3` | direct MINLP embedding |
| `[viz]` | `matplotlib` | plot helpers |
| `[uq-metrics]` | `uncertainty-toolbox` | calibration / sharpness metrics |
| `[steam]` | `iapws` | the steam-tables example |
| `[parity]` | `torch`, `scikit-learn` | numerical-equivalence tests vs upstream |
| `[dev]` | `pytest`, `hypothesis`, `ruff`, `mypy` | tooling |

## Quickstart

```python
import jax.numpy as jnp
from jax_ldt import HyperplaneTreeRegressor

X = jnp.asarray(...)
y = jnp.asarray(...)

model = HyperplaneTreeRegressor(max_depth=8, max_weight=2, num_terms=2)
model.fit(X, y)
yhat = model.predict(X_new)
```

`HyperplaneTreeRegressor` is the right default for ≥2D problems.
`LinearTreeRegressor` skips the hyperplane lift; use it for 1D inputs
or when you specifically want axis-aligned cuts.

## Training

Both regressors expose a scikit-learn-style `.fit(X, y) / .predict(X)`
interface. Inputs may be NumPy or JAX arrays; computation is
`float64` by default.

| param | default | effect |
|-------|---------|--------|
| `max_depth` | 32 | tree depth cap |
| `max_bins` | 10 (HT) / 25 (LMDT) | quantile cuts per feature |
| `min_samples_leaf` | 0.01 (frac) | min rows per leaf |
| `min_samples_split` | 6 | min rows to attempt a split |
| `ridge` | 1e-5 | ridge regularisation in leaf fits |
| `criterion` | `"mae"` | one of `mae` / `rmse` / `msle` / `max_abs` |
| `max_weight` (HT) | 1 | int weight cap in hyperplane enumeration |
| `num_terms` (HT) | 2 | features combined per oblique split |

Practical defaults: `max_depth=5–8`, `min_samples_leaf=15` for noisy
real data, and `num_terms=2`, `max_weight ∈ {1, 2}` for ≥6 features
(higher values blow up the lifted-feature count combinatorially).

The growth loop is a Python `while` over a queue of node ids (its
topology is data-dependent), but each split call dispatches to a
JIT-compiled kernel that vmaps a ridge fit over every candidate split
on a shape-stable `(N, n_aug)` mask tensor — one compile, then fast.

## Uncertainty quantification

Three composable methods, all available out of the box:

| method | what it gives you | when |
|---|---|---|
| `LinearPropagationUQ` | leaf-local Gaussian σ from the fitted linear model | cheap pointwise standard errors |
| `QuadraticUQ` | second-order correction across leaves | smoother σ near boundaries |
| `ConformalCalibrator` | distribution-free **prediction intervals with valid coverage** | when you need honest 1−α intervals |

The conformal calibrator supports **Mondrian (per-leaf) quantiles**
with a sparse-leaf fallback to the global quantile, so you get tight
intervals where data is dense without losing coverage in lonely
leaves:

```python
from jax_ldt import HyperplaneTreeRegressor, ConformalCalibrator

model = HyperplaneTreeRegressor(...).fit(X_train, y_train)
calib = ConformalCalibrator(
    alpha=0.1,                       # 90% intervals
    mondrian=True,                   # per-leaf quantile
    min_calibration_per_leaf=5,      # fallback threshold
).calibrate(model, X_cal, y_cal)
lo, hi = calib.predict_interval(X_test, model=model)
```

Convenience: `fit_with_conformal(model, X, y, calibration_size=0.2)`
splits internally and returns `(model, calib)`.

## Active learning

Batched, with `ask` / `tell` primitives and a `run` callback for
in-silico loops. Acquisitions include `UncertaintySampler`,
`MaxVariance`, `ExpectedImprovement`, and `ProbabilityOfImprovement`;
batchers include `TopKBatchSelector`,
`GreedyMaxMinBatchSelector` (the recommended default), and
`DiverseBatchSelector` (KMeans-clustered top-N).

```python
from jax_ldt import (
    ActiveLearner, HyperplaneTreeRegressor,
    ExpectedImprovement, GreedyMaxMinBatchSelector,
)

loop = ActiveLearner(
    model_factory=lambda: HyperplaneTreeRegressor(max_depth=5),
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    bounds=bounds, batch_size=8, seed=42,
)
loop.tell(X0, y0)

batch_id, X_batch = loop.ask()
loop.save("campaign.json")                       # crash-safe
y_partial = run_lab(X_batch[[0, 2, 5]])
loop.tell(batch_id, y_partial, indices=[0, 2, 5])  # partial replies OK
```

The campaign file persists data, RNG state, history, and pending
batches but **not** the trained tree (it is refit on `load`), so
campaigns survive internal layout changes. Index-aligned `tell`
accepts subsets of a batch — unanswered rows stay pending and can be
re-proposed in the next round.

## Export and MIP embedding

Trained trees are pytrees and serialise three ways:

```python
from jax_ldt import to_json, from_json, to_onnx

to_json(model, "tree.json")    # canonical, human-readable
to_onnx(model, "tree.onnx")    # for onnxruntime / OMLT (external)
```

For **direct MINLP optimisation** without Pyomo, embed in
[`discopt`](https://pypi.org/project/discopt/):

```python
import discopt
from jax_ldt.export import embed_in_discopt_model

m = discopt.Model("min-y")
x = m.continuous("x", shape=(n_features,), lb=lb, ub=ub)
y_expr = embed_in_discopt_model(model, m, x, big_m=20.0)
m.minimize(y_expr[0])
result = m.solve()
```

The adapter encodes the standard "GDP via big-M" pattern OMLT uses,
built directly with discopt primitives: lifted-feature equality
constraints, one binary `z_k` per leaf with `Σ z_k = 1`, big-M
routing for every internal split, and big-M-linearised bilinear leaf
regressions.

## Examples

The `examples/` directory contains six executed notebooks:

| notebook | covers |
|---|---|
| `01_quickstart.ipynb` | LMDT vs HT on Branin |
| `02_uncertainty.ipynb` | linprop / quadratic / Mondrian conformal with 90% coverage |
| `03_active_learning.ipynb` | EI loop on a 2D quadratic; save/resume |
| `04_mip_optimization.ipynb` | discopt MINLP minimising a learned tree |
| `05_onnx_export.ipynb` | ONNX round-trip to 1e-9 |
| `06_steam_tables.ipynb` | active-learning PHT surrogate against IAPWS-97 |

See [docs/](docs/) for the full guide: algorithm details, uncertainty
quantification, active-learning loop, and MIP export.

## License

MIT
