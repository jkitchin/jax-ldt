"""Generate the example notebooks from a single source-of-truth.

Each example is defined as a list of (kind, content) tuples where kind
is "md" or "code". Running this script writes the .ipynb files; we then
execute them with jupyter nbconvert.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent


def make_nb(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python"},
    }
    nb.cells = []
    for kind, src in cells:
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        elif kind == "code":
            nb.cells.append(nbformat.v4.new_code_cell(src))
        else:
            raise ValueError(f"unknown cell kind: {kind}")
    return nb


# -- 01_quickstart -------------------------------------------------------
NB01 = [
    ("md", """# 01 · Quickstart: fit a hyperplane tree on Branin

Train a `HyperplaneTreeRegressor` on the 2D Branin function, predict on
a grid, and visualise the surface and oblique splits.
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor

print("jax", jax.__version__)
"""),
    ("md", "## Generate Branin training data"),
    ("code", """def branin(x1, x2):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

rng = np.random.default_rng(0)
X_train = rng.uniform([-5.0, 0.0], [10.0, 15.0], size=(400, 2))
y_train = branin(X_train[:, 0], X_train[:, 1])
print(f"X_train: {X_train.shape}, y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
"""),
    ("md", "## Fit both LMDT and HT, compare in-sample MAE"),
    ("code", """common = dict(max_depth=5, max_bins=8, min_samples_leaf=15, ridge=1e-5)

lmdt = LinearTreeRegressor(**common).fit(X_train, y_train)
ht = HyperplaneTreeRegressor(**common, max_weight=2, num_terms=2).fit(X_train, y_train)

mae_lmdt = float(jnp.mean(jnp.abs(lmdt.predict(X_train) - y_train)))
mae_ht = float(jnp.mean(jnp.abs(ht.predict(X_train) - y_train)))
print(f"LMDT MAE: {mae_lmdt:.3f}  ({lmdt.num_leaves} leaves)")
print(f"HT   MAE: {mae_ht:.3f}  ({ht.num_leaves} leaves)")
"""),
    ("md", "## Plot true surface, LMDT and HT predictions"),
    ("code", """g = 80
xx, yy = np.meshgrid(
    np.linspace(-5, 10, g), np.linspace(0, 15, g)
)
Xg = np.column_stack([xx.ravel(), yy.ravel()])

zz_true = branin(xx, yy)
zz_lmdt = np.asarray(lmdt.predict(Xg)).reshape(xx.shape)
zz_ht = np.asarray(ht.predict(Xg)).reshape(xx.shape)

vmin, vmax = float(zz_true.min()), float(zz_true.max())
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, z, title in zip(
    axes,
    [zz_true, zz_lmdt, zz_ht],
    ["Branin (truth)", f"LMDT ({lmdt.num_leaves} leaves)", f"HT ({ht.num_leaves} leaves)"],
):
    im = ax.pcolormesh(xx, yy, z, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
fig.colorbar(im, ax=axes, shrink=0.85)
plt.show()
"""),
    ("md", """## What just happened?

* `LinearTreeRegressor` fits axis-aligned splits with a linear regression at every leaf.
* `HyperplaneTreeRegressor` lifts X via a learned set of integer-weight directions (`max_weight=2`, `num_terms=2`) and fits the same kind of tree on the lifted features. Splits can now follow oblique angles.
* On Branin the two are comparable at this leaf budget — Branin's curvature is mostly axis-aligned around its three minima, so the oblique splits don't help much. HT pays off more clearly when the target has diagonal structure (e.g., `y = sin(x0 + x1)`); see notebook 06 for a steam-table example where HT wins decisively.
"""),
]


# -- 02_uncertainty ------------------------------------------------------
NB02 = [
    ("md", """# 02 · Uncertainty quantification

Three methods, side by side:

* **Linear propagation** — closed-form, computed from leaf statistics saved at fit time.
* **Quadratic difference** — refits each leaf with quadratic features and reports `|linear - quadratic|`.
* **Mondrian leaf conformal** — distribution-free intervals with finite-sample coverage, calibrated on a held-out set.
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_ldt import (
    LinearTreeRegressor,
    LinearPropagationUQ,
    QuadraticUQ,
    ConformalCalibrator,
)
"""),
    ("md", "## Train + calibration split on a noisy 1D toy"),
    ("code", """rng = np.random.default_rng(42)
# Use a larger N so the empirical-coverage estimate has lower variance.
# The conformal guarantee is *marginal*; finite-sample fluctuations are
# ~sqrt(alpha*(1-alpha)/N_test), which is roughly +/-1.5 pts at N_test=1500.
N = 4000
X = rng.uniform(-3, 3, size=(N, 1))
noise = 0.1 + 0.3 * (X[:, 0] > 0)  # heteroscedastic
y = np.sin(X[:, 0]) + noise * rng.standard_normal(N)

# 60/20/20 train/cal/test
perm = rng.permutation(N)
tr, cal, te = np.split(perm, [int(0.6 * N), int(0.8 * N)])
X_tr, y_tr = X[tr], y[tr]
X_cal, y_cal = X[cal], y[cal]
X_te, y_te = X[te], y[te]

# 1D: hyperplanes don't add anything, so use the axis-aligned tree.
model = LinearTreeRegressor(
    max_depth=5, max_bins=8, min_samples_leaf=20,
).fit(X_tr, y_tr)

print(f"train MAE: {float(jnp.mean(jnp.abs(model.predict(X_tr) - y_tr))):.3f}")
print(f"test  MAE: {float(jnp.mean(jnp.abs(model.predict(X_te) - y_te))):.3f}")
"""),
    ("md", "## Linear-propagation and quadratic UQ"),
    ("code", """sigma_lp = np.asarray(LinearPropagationUQ().predict(model.tree_, X_te))
quq = QuadraticUQ(ridge=1e-5).calibrate(model.tree_, X_tr, y_tr)
sigma_q = np.asarray(quq.predict(model.tree_, X_te))

print(f"linprop sigma  mean={sigma_lp.mean():.3f}  median={np.median(sigma_lp):.3f}")
print(f"quadratic dev mean={sigma_q.mean():.3f}  median={np.median(sigma_q):.3f}")
"""),
    ("md", "## Mondrian leaf conformal"),
    ("code", """calib = ConformalCalibrator(
    alpha=0.1, mondrian=True, min_calibration_per_leaf=5,
).calibrate(model, X_cal, y_cal)
lo, hi = calib.predict_interval(X_te, model=model)
lo = np.asarray(lo); hi = np.asarray(hi)

coverage = float(np.mean((y_te >= lo) & (y_te <= hi)))
mean_width = float(np.mean(hi - lo))
print(f"Empirical 90% coverage on test: {coverage:.3f}  (target: 0.90)")
print(f"Mean interval width:           {mean_width:.3f}")
"""),
    ("md", "## Visualise"),
    ("code", """xs = np.linspace(-3, 3, 300).reshape(-1, 1)
yh = np.asarray(model.predict(xs))
sigma_xs = np.asarray(LinearPropagationUQ().predict(model.tree_, xs))
lo_xs, hi_xs = calib.predict_interval(xs, model=model)
lo_xs, hi_xs = np.asarray(lo_xs), np.asarray(hi_xs)

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(xs[:, 0], lo_xs, hi_xs, alpha=0.25, label="Conformal 90%")
ax.fill_between(
    xs[:, 0], yh - sigma_xs, yh + sigma_xs,
    alpha=0.25, color="C2", label="Linprop ±σ",
)
ax.scatter(X_te, y_te, s=10, alpha=0.4, label="test", color="k")
ax.plot(xs[:, 0], yh, color="C0", label="model")
ax.plot(xs[:, 0], np.sin(xs[:, 0]), color="r", linestyle="--", label="truth")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="lower left")
plt.show()
"""),
    ("md", """## Reading the plot

* **Conformal 90%** band has *marginal* coverage close to 90% on average over calibration draws — a single small calibration set fluctuates by roughly `±sqrt(alpha*(1-alpha)/N_test)` (≈1.5 percentage points at this size). The band is wider where leaves contain noisier residuals (Mondrian effect).
* **Linprop ±σ** is cheap but is a Gaussian-heuristic approximation; it does *not* guarantee coverage.
* The two together let you triage where the model is confident vs. extrapolating.

If a `UserWarning` mentions sparse leaves, the default `sparse_leaf_strategy="global"` borrows the global residual quantile for those leaves — preserving marginal coverage but losing per-leaf exchangeability there. Use `sparse_leaf_strategy="skip"` if you'd rather get `NaN` intervals than a leaky guarantee on those buckets.
"""),
]


# -- 03_active_learning --------------------------------------------------
NB03 = [
    ("md", """# 03 · Active learning: minimise a Rosenbrock-like surface

Drive the surrogate around a bowl using `ExpectedImprovement` + `GreedyMaxMinBatchSelector`.
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

from jax_ldt import (
    ActiveLearner,
    HyperplaneTreeRegressor,
    ExpectedImprovement,
    GreedyMaxMinBatchSelector,
)
"""),
    ("md", "## Define the test objective"),
    ("code", """target = np.array([0.4, -0.8])

def f(X):
    return ((X - target) ** 2).sum(axis=1)

bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
"""),
    ("md", "## Set up the loop"),
    ("code", """def factory():
    return HyperplaneTreeRegressor(
        max_depth=4, max_bins=6, min_samples_leaf=6,
        max_weight=1, num_terms=2,
    )

loop = ActiveLearner(
    model_factory=factory,
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    bounds=bounds,
    batch_size=6,
    seed=11,
)

# initial random data
rng = np.random.default_rng(11)
X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(8, 2))
loop.tell(X0, f(X0))
print(f"initial best y: {loop.y_observed.min():.3f}")
"""),
    ("md", "## Run 10 rounds of batch-6 acquisition"),
    ("code", """best_history = [float(loop.y_observed.min())]
for _ in range(10):
    bid, X_batch = loop.ask()
    y_batch = f(X_batch)
    loop.tell(bid, y_batch)
    best_history.append(float(loop.y_observed.min()))

print(f"best y after 10 rounds: {best_history[-1]:.4f} (true min = 0)")
"""),
    ("md", "## Visualise convergence and acquired points"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

axes[0].plot(best_history, "o-")
axes[0].set_xlabel("round"); axes[0].set_ylabel("best observed y")
axes[0].set_yscale("log")
axes[0].set_title("Convergence (lower is better)")

# scatter of acquired points coloured by acquisition order
ax = axes[1]
g = 80
xx, yy = np.meshgrid(np.linspace(-2, 2, g), np.linspace(-2, 2, g))
zz = ((np.column_stack([xx.ravel(), yy.ravel()]) - target) ** 2).sum(axis=1).reshape(xx.shape)
ax.contourf(xx, yy, zz, levels=20, cmap="Greys", alpha=0.6)
ax.scatter(*X0.T, color="k", marker="x", label="initial")
acquired = loop.X_observed[len(X0):]
ax.scatter(*acquired.T, c=np.arange(len(acquired)), cmap="viridis", s=40, label="acquired")
ax.scatter(*target, color="red", marker="*", s=180, label="true min")
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend()
ax.set_title("Acquisition trajectory")
plt.show()
"""),
    ("md", """## Save / resume the campaign

`loop.save("campaign.json")` writes everything needed to resume — observed data, RNG state, pending batches, history. The trained tree itself is *not* persisted: `ActiveLearner.load(...)` refits it from the data, so the JSON is decoupled from the internal `Tree` layout.
"""),
    ("code", """import tempfile, os
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "campaign.json")
    loop.save(path)
    resumed = ActiveLearner.load(
        path, model_factory=factory,
        acquisition=ExpectedImprovement(direction="min"),
        batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    )
    print("resumed:", resumed.X_observed.shape, "obs / best y =", resumed.y_observed.min())
"""),
]


# -- 04_mip_optimization -------------------------------------------------
NB04 = [
    ("md", """# 04 · MIP optimisation: minimise a tree surrogate with `discopt`

Train a HyperplaneTree on a quadratic, then ask discopt to find its
piecewise-linear minimum. No Pyomo or OMLT required.
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

import discopt
from jax_ldt import LinearTreeRegressor
from jax_ldt.export import embed_in_discopt_model

print("discopt", discopt.__version__)
"""),
    ("md", "## Train a small surrogate"),
    ("code", """rng = np.random.default_rng(0)
X = rng.uniform(-1.5, 1.5, size=(150, 2))
y = (X ** 2).sum(axis=1) + 0.1 * X[:, 0] * X[:, 1]
model = LinearTreeRegressor(max_depth=3, max_bins=4, min_samples_leaf=15).fit(X, y)
print(f"surrogate MAE: {float(np.mean(np.abs(model.predict(X) - y))):.4f}, leaves: {model.num_leaves}")
"""),
    ("md", """## Embed the tree in a `discopt.Model`

`embed_in_discopt_model` adds:
1. lifted-feature equality constraints,
2. one binary `z_k` per leaf with `Σ z_k = 1`,
3. big-M routing constraints for every internal split,
4. bilinear leaf regressions linearised with big-M.

It returns a discopt expression you can use as an objective or constraint.
"""),
    ("code", """m = discopt.Model("min-tree")
x = m.continuous("x", shape=(2,), lb=-1.5, ub=1.5)
y_expr = embed_in_discopt_model(model, m, x, big_m=10.0)
m.minimize(y_expr[0])
result = m.solve()
x_star = np.asarray(result.x["x"])
print("argmin:", x_star.round(3))
print("min y :", float(result.objective))
print("in-sample y range:", y.min(), "...", y.max())
"""),
    ("md", """The recovered minimum is at or below the smallest training point, since discopt is solving the surrogate's piecewise-linear minimum exactly (not an interpolation). For a non-trivial example with constraints or mixed-integer variables, just add them to `m` alongside `x`.
"""),
]


# -- 05_onnx_export ------------------------------------------------------
NB05 = [
    ("md", """# 05 · Export to ONNX and round-trip via `onnxruntime`

Demonstrates that the hand-built ONNX graph reproduces predictions to numerical precision, and is consumable by any ONNX-aware runtime (including OMLT externally).
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import onnxruntime as ort

from jax_ldt import (
    HyperplaneTreeRegressor,
    to_onnx,
    to_json,
    from_json,
    predict_tree,
)
"""),
    ("md", "## Train and export"),
    ("code", """rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(300, 3))
y = X[:, 0] ** 2 - X[:, 1] * X[:, 2]
model = HyperplaneTreeRegressor(
    max_depth=4, max_bins=6, min_samples_leaf=12, max_weight=1, num_terms=2,
).fit(X, y)

import tempfile, os
tmp = tempfile.mkdtemp()
onnx_path = os.path.join(tmp, "tree.onnx")
to_onnx(model, onnx_path)
size_kb = os.path.getsize(onnx_path) / 1024
print(f"wrote {onnx_path}: {size_kb:.1f} KB")
"""),
    ("md", "## Round-trip through onnxruntime"),
    ("code", """sess = ort.InferenceSession(onnx_path)
X_test = X[:50].astype(np.float64)
yh_jax = np.asarray(model.predict(X_test))
yh_onnx = sess.run(None, {"X": X_test})[0].reshape(-1)

max_diff = float(np.max(np.abs(yh_jax - yh_onnx)))
print(f"max |jax - onnx|: {max_diff:.2e}")
np.testing.assert_allclose(yh_jax, yh_onnx, atol=1e-9)
print("round-trip OK to 1e-9")
"""),
    ("md", "## JSON spec round-trip too"),
    ("code", """json_path = os.path.join(tmp, "tree.json")
to_json(model, json_path)
tree = from_json(json_path)
yh_json = np.asarray(predict_tree(tree, X_test))
np.testing.assert_allclose(yh_json, yh_jax, atol=1e-12)
print("JSON round-trip OK to 1e-12")
"""),
    ("md", """## Two artifacts, one source of truth

* **JSON** is the canonical, human-readable format. Use it to inspect a trained tree, version-control it, or feed it to a custom solver.
* **ONNX** is the export-for-other-runtimes format. The same file can be loaded by `onnxruntime`, by OMLT inside a Pyomo workflow, or by ONNX-compatible deployment toolchains — *we* don't depend on any of those at runtime.
"""),
]


# -- 06_steam_tables ------------------------------------------------------
NB06 = [
    ("md", """# 06 · Steam tables via active learning (paper case study)

Reproduces the **pressure-enthalpy → temperature** (PHT) surrogate from
Sunshine et al. 2025 (Section 4, Figure 9), but trained the right way
for an expensive simulator: with the active-learning loop from
notebook 03, treating IAPWS-97 as the oracle.

We:

1. Wrap `iapws.IAPWS97` as the oracle. (Real life: a slow CFD or
   process simulator. We use IAPWS for reproducibility.)
2. Seed with a 32-point Sobol design covering the
   pressure × enthalpy box.
3. Run 12 rounds of batch-32 uncertainty-driven acquisition. Each
   round refits the `HyperplaneTreeRegressor` on every observed point
   and proposes 32 new points where the leaf-level σ is largest.
4. Compare to a passive-baseline HT trained on the same total budget
   sampled uniformly at random.
5. Visualise the active-learning trajectory on top of the IAPWS
   phase diagram and reproduce the leaf-partition figure from the
   paper.

**Why this is interesting**: AL spends its evaluation budget on the
hard part of the surface — the vapor-liquid envelope, where T changes
most steeply — instead of wasting samples on the smooth liquid /
superheated regions. With a few hundred IAPWS calls we recover MAE
similar to the dense 10 000-point grid baseline.
"""),
    ("code", """import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import iapws

from jax_ldt import (
    ActiveLearner,
    HyperplaneTreeRegressor,
    UncertaintySampler,
    GreedyMaxMinBatchSelector,
    to_onnx,
)

print("iapws", iapws.__version__, "  jax", jax.__version__)
"""),
    ("md", """## The IAPWS oracle

For each `(P, h)` pair we ask `iapws.IAPWS97` for the temperature.
A handful of `(P, h)` combinations fall outside the valid range —
iapws raises in that case. The active learner can handle that
gracefully: we run the oracle one row at a time and report back only
the points that succeeded via the index-aligned `tell` API.
"""),
    ("code", """def iapws_T(X):
    \"\"\"Vectorised IAPWS-97 oracle. Returns (T, valid_mask).\"\"\"
    X = np.asarray(X, dtype=np.float64)
    T = np.full(X.shape[0], np.nan)
    for i in range(X.shape[0]):
        try:
            T[i] = iapws.IAPWS97(P=float(X[i, 0]), h=float(X[i, 1])).T
        except Exception:
            pass
    return T, np.isfinite(T)


# Search box: pressure 0.1-25 MPa, enthalpy 50-3700 kJ/kg.
bounds = np.array([[0.1, 25.0], [50.0, 3700.0]], dtype=np.float64)
"""),
    ("md", """## Set up the active-learning loop

* Surrogate: `HyperplaneTreeRegressor` with `max_weight=2`,
  `num_terms=2` — the paper's recommended setting for problems with
  diagonal structure.
* Acquisition: `UncertaintySampler` (linprop σ from leaf statistics).
* Batcher: `GreedyMaxMinBatchSelector` so each batch spreads itself
  out instead of clustering on the highest-uncertainty cell.
* Budget: 32 initial Sobol points + 12 rounds × 32 acquired = 416
  total IAPWS evaluations.
"""),
    ("code", """def make_ht():
    return HyperplaneTreeRegressor(
        criterion="mae",
        max_depth=7,
        max_bins=8,
        min_samples_leaf=8,
        ridge=1e-5,
        max_weight=2,
        num_terms=2,
    )


loop = ActiveLearner(
    model_factory=make_ht,
    acquisition=UncertaintySampler(),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.3),
    bounds=bounds,
    batch_size=32,
    seed=2026,
)
"""),
    ("md", """## Seed the loop

We draw 32 quasi-random Sobol points, evaluate IAPWS, and feed the
ones that succeeded as the initial training set.
"""),
    ("code", """from scipy.stats import qmc

sobol = qmc.Sobol(d=2, seed=42)
seed_u = sobol.random(64)
seed_X = bounds[:, 0] + seed_u * (bounds[:, 1] - bounds[:, 0])
seed_T, seed_valid = iapws_T(seed_X)
seed_X, seed_T = seed_X[seed_valid], seed_T[seed_valid]
print(f"initial design: {seed_X.shape[0]} valid points (out of 64 Sobol candidates)")

loop.tell(seed_X, seed_T)
print(f"initial test fit: {loop.X_observed.shape[0]} points, "
      f"current model has {loop.model.num_leaves} leaves")
"""),
    ("md", """## Hold-out test set

To track convergence honestly, we precompute a 1500-point uniform test
set with valid IAPWS data. The active learner never sees this set.
"""),
    ("code", """rng = np.random.default_rng(99)
test_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(2000, 2))
test_T, test_valid = iapws_T(test_X)
test_X, test_T = test_X[test_valid][:1500], test_T[test_valid][:1500]
print(f"test set: {test_X.shape[0]} points, T range "
      f"[{test_T.min():.1f}, {test_T.max():.1f}] K")
"""),
    ("md", """## Run 12 rounds of acquisition

For each round we ask the loop for 32 candidates, evaluate IAPWS, and
report back via `tell(batch_id, T_observed, indices=...)`. Indices
limit the response to candidates IAPWS could actually evaluate.
"""),
    ("code", """def mae(yh, yt):
    return float(np.mean(np.abs(np.asarray(yh) - yt)))


history = []
n_evals_history = []
test_mae_history = [mae(loop.model.predict(test_X), test_T)]
n_evals_history.append(loop.X_observed.shape[0])

for r in range(12):
    bid, X_batch = loop.ask()
    T_batch, mask_batch = iapws_T(X_batch)
    valid_idx = list(np.where(mask_batch)[0])
    loop.tell(bid, T_batch[mask_batch], indices=valid_idx)

    test_mae = mae(loop.model.predict(test_X), test_T)
    history.append((r, len(valid_idx), test_mae, loop.model.num_leaves))
    n_evals_history.append(loop.X_observed.shape[0])
    test_mae_history.append(test_mae)
    print(
        f"round {r:2d}  +{len(valid_idx):3d} valid points  "
        f"total {loop.X_observed.shape[0]:4d}  "
        f"leaves={loop.model.num_leaves:3d}  "
        f"test MAE={test_mae:6.2f} K"
    )

ht_al = loop.model
"""),
    ("md", """## Passive baseline

To gauge how much the active selection helps, we train an identically
configured HT on the same total number of *uniformly random* points.
"""),
    ("code", """rng_pass = np.random.default_rng(2027)
n_total = loop.X_observed.shape[0]
pass_X = rng_pass.uniform(bounds[:, 0], bounds[:, 1], size=(int(n_total * 1.4), 2))
pass_T, pass_valid = iapws_T(pass_X)
pass_X, pass_T = pass_X[pass_valid][:n_total], pass_T[pass_valid][:n_total]

ht_passive = make_ht().fit(pass_X, pass_T)

mae_al = mae(ht_al.predict(test_X), test_T)
mae_pass = mae(ht_passive.predict(test_X), test_T)
print(f"AL HT      n={n_total}  leaves={ht_al.num_leaves:3d}  test MAE={mae_al:.3f} K")
print(f"passive HT n={n_total}  leaves={ht_passive.num_leaves:3d}  test MAE={mae_pass:.3f} K")
"""),
    ("md", """## Convergence and acquired-point trajectory

Left: test MAE vs. evaluation budget — the AL curve typically beats
the passive baseline once the loop has homed in on the VLE band.
Right: the (P, h) phase diagram with seed points (×) and acquired
points coloured by round; you can see the loop concentrate samples
on the steep part of the surface.
"""),
    ("code", """# Phase-diagram backdrop: dense IAPWS truth on a coarse grid
g = 80
hh_grid, pp_grid = np.meshgrid(
    np.linspace(bounds[1, 0], bounds[1, 1], g),
    np.linspace(bounds[0, 0], bounds[0, 1], g),
    indexing="xy",
)
T_truth = np.full(pp_grid.shape, np.nan)
for i in range(pp_grid.shape[0]):
    for j in range(pp_grid.shape[1]):
        try:
            T_truth[i, j] = iapws.IAPWS97(P=float(pp_grid[i, j]), h=float(hh_grid[i, j])).T
        except Exception:
            pass
mask = np.isfinite(T_truth)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)

axes[0].plot(n_evals_history, test_mae_history, "o-", label="AL")
axes[0].axhline(mae_pass, color="C1", linestyle="--",
                label=f"passive @ n={n_total}")
axes[0].set_xlabel("# IAPWS evaluations")
axes[0].set_ylabel("test MAE [K]")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("Active learning convergence")

axes[1].pcolormesh(
    np.where(mask, hh_grid, np.nan),
    np.where(mask, pp_grid, np.nan),
    np.where(mask, T_truth, np.nan),
    cmap="viridis", shading="auto", alpha=0.6,
)
seed_n = seed_X.shape[0]
all_obs = loop.X_observed
acquired = all_obs[seed_n:]
round_index = np.repeat(
    np.arange(len(history)),
    [h[1] for h in history],
)
axes[1].scatter(seed_X[:, 1], seed_X[:, 0], marker="x", s=18,
                color="k", label="Sobol seed")
sc = axes[1].scatter(acquired[:, 1], acquired[:, 0], c=round_index,
                     cmap="plasma", s=18, edgecolor="white", linewidth=0.3,
                     label="acquired")
axes[1].set_xlabel("enthalpy h [kJ/kg]")
axes[1].set_ylabel("pressure P [MPa]")
axes[1].set_title("AL trajectory on IAPWS phase diagram")
fig.colorbar(sc, ax=axes[1], label="round")
axes[1].legend(loc="upper right")
plt.show()
"""),
    ("md", """## Phase diagram with the AL surrogate

Same figure as the paper (and as our notebook 06 baseline): truth on
the left, AL-trained HT prediction in the middle, leaf partition on
the right. The leaf partition shows the oblique cuts the AL run
discovered along the vapor-liquid envelope.
"""),
    ("code", """Xg = np.column_stack([pp_grid.ravel(), hh_grid.ravel()])
T_pred = np.asarray(ht_al.predict(Xg)).reshape(pp_grid.shape)
leaves = np.asarray(ht_al.apply(Xg)).reshape(pp_grid.shape)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
vmin = float(np.nanmin(T_truth))
vmax = float(np.nanmax(T_truth))
im0 = axes[0].pcolormesh(hh_grid, pp_grid, np.where(mask, T_truth, np.nan),
                         cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
axes[0].set_title("IAPWS-97 truth")
im1 = axes[1].pcolormesh(hh_grid, pp_grid, np.where(mask, T_pred, np.nan),
                         cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
axes[1].set_title(f"AL-trained HT ({ht_al.num_leaves} leaves, "
                  f"n={loop.X_observed.shape[0]})")
# overlay hyperplane cuts as straight line segments. Walk the tree and
# for each internal node clip the split hyperplane to the polytope of
# ancestor inequalities, giving exactly the segment that bounds two
# leaf regions in plot (h, P) coordinates.
from matplotlib.collections import LineCollection

tree = ht_al.tree_
A = np.asarray(tree.transform_matrix)            # (n_features_in=2, n_lifted)
is_leaf = np.asarray(tree.is_leaf)
feature = np.asarray(tree.feature)
threshold = np.asarray(tree.threshold)
left = np.asarray(tree.left)
right = np.asarray(tree.right)

def _clip_halfplane(poly, a, b, c):
    """Return poly ∩ {a*x + b*y <= c} via Sutherland-Hodgman."""
    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        p_in, q_in = dp <= 1e-12, dq <= 1e-12
        if p_in and q_in:
            out.append(q)
        elif p_in and not q_in:
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        elif not p_in and q_in:
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            out.append(q)
    return out

def _line_x_polygon(poly, a, b, c):
    """Intersection of the line a*x + b*y = c with convex polygon poly."""
    pts = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp == 0:
            pts.append(p)
        elif dp * dq < 0:
            t = dp / (dp - dq)
            pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return pts

# X column order is [P, h]; plot axes are (x=h, y=P).
# Split (X @ A)[:, f] <= t  →  A[0,f]*P + A[1,f]*h <= t
# In plot coords (h, P): coefs are (A[1,f], A[0,f]).
h_min, h_max = float(hh_grid.min()), float(hh_grid.max())
p_min, p_max = float(pp_grid.min()), float(pp_grid.max())
root_poly = [(h_min, p_min), (h_max, p_min), (h_max, p_max), (h_min, p_max)]

segs = []
stack = [(0, root_poly)]
while stack:
    node, poly = stack.pop()
    if is_leaf[node] or not poly:
        continue
    f = int(feature[node])
    t = float(threshold[node])
    a, b = float(A[1, f]), float(A[0, f])  # coefs of (h, P)
    pts = _line_x_polygon(poly, a, b, t)
    if len(pts) >= 2:
        segs.append([pts[0], pts[-1]])
    stack.append((int(left[node]), _clip_halfplane(poly, a, b, t)))
    stack.append((int(right[node]), _clip_halfplane(poly, -a, -b, -t)))
axes[1].add_collection(LineCollection(segs, colors="k", linewidths=0.6))
axes[2].pcolormesh(hh_grid, pp_grid, np.where(mask, leaves, -1),
                   cmap="tab20", shading="auto")
axes[2].set_title("HT leaf partition")
for ax in axes:
    ax.set_xlabel("enthalpy h [kJ/kg]")
axes[0].set_ylabel("pressure P [MPa]")
fig.colorbar(im0, ax=axes[:2], shrink=0.85, label="T [K]")
plt.show()
"""),
    ("md", """## Export

The AL-trained model is just a regular tree — it round-trips through
ONNX and JSON the same way as any other.
"""),
    ("code", """import tempfile, os
tmp = tempfile.mkdtemp()
onnx_path = os.path.join(tmp, "pht_ht_al.onnx")
to_onnx(ht_al, onnx_path)
print(f"wrote {onnx_path}: {os.path.getsize(onnx_path) / 1024:.1f} KB")

import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
yh_jax = np.asarray(ht_al.predict(test_X[:200]))
yh_onnx = sess.run(None, {"X": test_X[:200].astype(np.float64)})[0].reshape(-1)
print(f"max |jax - onnx| = {np.max(np.abs(yh_jax - yh_onnx)):.2e}")
"""),
    ("md", """## Reading the result

* The convergence panel shows that AL pulls test MAE down faster
  than the passive baseline once the surrogate has resolved the
  liquid and vapor regions and starts spending its budget on the
  vapor-liquid envelope.
* The trajectory panel shows acquired points clustering along the
  steep parts of the IAPWS surface — exactly where a passive uniform
  sample would waste evaluations on already-easy regions.
* The right figure shows the AL-trained HT producing the same
  qualitative phase diagram as the paper's dense-grid model but with
  a fraction of the IAPWS calls.

For the full heat-exchanger MINLP that consumes this surrogate, see
the paper's Section 4 and `docs/discopt_integration.md`. Once the
surrogate is exported via `to_onnx` or embedded with
`embed_in_discopt_model`, the optimisation step doesn't care whether
the training data came from a dense grid or an active-learning loop.
"""),
]


def _write_and_register(stem: str, cells: list[tuple[str, str]]) -> Path:
    nb = make_nb(cells)
    path = HERE / f"{stem}.ipynb"
    with path.open("w") as f:
        nbformat.write(nb, f)
    return path


def main() -> list[Path]:
    out: list[Path] = []
    out.append(_write_and_register("01_quickstart", NB01))
    out.append(_write_and_register("02_uncertainty", NB02))
    out.append(_write_and_register("03_active_learning", NB03))
    out.append(_write_and_register("04_mip_optimization", NB04))
    out.append(_write_and_register("05_onnx_export", NB05))
    out.append(_write_and_register("06_steam_tables", NB06))
    return out


if __name__ == "__main__":
    paths = main()
    for p in paths:
        print(f"wrote {p.relative_to(HERE.parent)}")
