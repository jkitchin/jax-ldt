# Quickstart

## Install

```bash
pip install -e ".[dev,onnx]"
```

The `[onnx]` extra adds `onnx` and `onnxruntime` for export and
round-trip verification. Other optional extras:

- `[discopt]` — the discopt adapter and DoE acquisition.
- `[viz]` — matplotlib helpers.
- `[uq-metrics]` — `uncertainty-toolbox` for calibration metrics.

## Fit a hyperplane tree

```python
import jax.numpy as jnp
import numpy as np

from jax_ldt import HyperplaneTreeRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(400, 2))
y = np.sin(X[:, 0]) + 0.1 * X[:, 1] ** 2 + 0.05 * rng.standard_normal(400)

model = HyperplaneTreeRegressor(
    max_depth=5, max_bins=8, min_samples_leaf=15,
    max_weight=1, num_terms=2,
)
model.fit(X, y)

X_grid = np.linspace(-3, 3, 50).reshape(-1, 1) * np.ones((1, 2))
y_pred = model.predict(X_grid)
```

## Add uncertainty

Linear-propagation uncertainty is computed from leaf-level statistics
saved during fit; no extra calibration step needed:

```python
from jax_ldt import LinearPropagationUQ

sigma = LinearPropagationUQ().predict(model.tree_, X_grid)
```

For distribution-free intervals, calibrate on held-out data:

```python
from jax_ldt import ConformalCalibrator

X_train, X_cal = X[:300], X[300:]
y_train, y_cal = y[:300], y[300:]
model.fit(X_train, y_train)

calib = ConformalCalibrator(alpha=0.1, mondrian=True).calibrate(
    model, X_cal, y_cal
)
lo, hi = calib.predict_interval(X_grid, model=model)  # 90% intervals
```

## Save / load

```python
from jax_ldt import to_json, from_json, to_onnx, predict_tree

# Solver-neutral JSON
to_json(model, "tree.json")
tree = from_json("tree.json")
y_again = predict_tree(tree, X_grid)

# ONNX (consumable by OMLT, onnxruntime, …)
to_onnx(model, "tree.onnx")
```

## Next steps

- [Active learning loop](active_learning.md): `ask` for the next batch
  of experiments, `tell` it the results.
- [MIP embedding](mip_export.md): solve `min y(x)` over the trained
  tree without Pyomo.
- [discopt integration](discopt_integration.md): full MINLP example.
