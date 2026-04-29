# MIP / surrogate-embedding export

A trained jax-ldt tree exports to two solver-neutral formats and one
direct embedding into discopt.

## JSON spec (canonical)

```python
from jax_ldt import to_json, from_json, predict_tree
to_json(model, "tree.json")
tree = from_json("tree.json")
y_again = predict_tree(tree, X)  # functional predict on the raw pytree
```

Schema (v1):

```json
{
  "version": 1,
  "kind": "linear_tree" | "hyperplane_tree",
  "n_features_in": 2,
  "n_features_transformed": 6,
  "n_targets": 1,
  "transform_matrix": [[...]],
  "categorical_features": [],
  "linear_features": [0, 1, 2, 3, 4, 5],
  "tree": {
    "node_count": 31,
    "is_leaf":   [false, false, true, ...],
    "feature":   [3, 1, -1, ...],
    "threshold": [2.5, -1.0, null, ...],
    "left":      [1, 3, -1, ...],
    "right":     [2, 4, -1, ...],
    "leaf_params": [[[...]]]
  },
  "uq": {
    "method": "linprop",
    "leaf_n":     [...],
    "leaf_x_mean":[[...]],
    "leaf_x_var": [[...]],
    "leaf_mse":   [[...]]
  }
}
```

This is the source of truth: any other export can be regenerated from
the JSON.

## ONNX

```python
from jax_ldt import to_onnx
to_onnx(model, "tree.onnx")
```

The graph encodes:

```
X → MatMul(transform_matrix) → X_t
X_t → augment with bias → X_aug
X_aug, X_t → routed leaf id (Gather + Where, unrolled to max depth)
leaf id → Gather(leaf_params) → per-row params
X_aug · params → Y
```

We don't use the ONNX-ML `TreeEnsembleRegressor` op because it only
encodes constant-leaf trees; ours have linear leaf models. Instead we
synthesise the routing in core ONNX ops, giving a graph any standard
runtime can execute.

### Round-trip with onnxruntime

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("tree.onnx")
y_onnx = sess.run(None, {"X": X.astype(np.float64)})[0]
np.testing.assert_allclose(y_onnx.reshape(-1), model.predict(X), atol=1e-8)
```

The `atol=1e-8` bound is empirical (G-14): the ONNX graph compounds
float64 rounding through `MatMul → Gather → ReduceSum`, which can drift
a few ULPs from the JAX reference even on identical hardware. For
shallow trees (depth ≤ 4) drift typically stays below `1e-9`; deeper
trees (depth 8 with hyperplane routing) widen the bound to roughly
`1e-7`. None of these are large enough to matter for downstream
surrogate-optimisation use, but they are above the strict `1e-9`
threshold one might first reach for.

### OMLT consumption

The same ONNX file can be loaded by [OMLT](https://github.com/cog-imperial/OMLT)
in any environment that has Pyomo. We document the recipe here but do
**not** depend on `pyomo` or `omlt` at runtime.

```python
# In a separate environment with omlt / pyomo installed:
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io import load_onnx_neural_network

m = pyo.ConcreteModel()
m.surrogate = OmltBlock()
network = load_onnx_neural_network("tree.onnx")
m.surrogate.build_formulation(...)
```

(See OMLT docs for full details; the surrogate block accepts standard
ONNX neural-network-style graphs, which our tree's routing graph is.)

## discopt embedding (no Pyomo)

For full hybrid (split + linear-leaf) MINLP embedding without leaving
Python, use the discopt adapter:

```python
import discopt
from jax_ldt.export import embed_in_discopt_model

m = discopt.Model("tree-opt")
x = m.continuous("x", shape=(n_features,), lb=lb, ub=ub)
y_expr = embed_in_discopt_model(model, m, x)
m.minimize(y_expr[0])
result = m.solve()
print("argmin:", result.x, "min:", result.objective)
```

The adapter:

1. Adds lifted-feature variables `x'_j = Σ_i A[i,j] x_i` as linear
   equalities.
2. Adds one binary `z_k` per leaf and constrains `Σ z_k = 1`.
3. Adds big-M routing constraints for every internal node, ensuring
   the active leaf is consistent with each split.
4. Adds auxiliary variables `v_k = z_k · (β_k · x' + α_k)` linearised
   with big-M, returning `Σ v_k` as the surrogate output.

This is the standard "GDP via big-M" pattern used by OMLT internally,
built directly with discopt primitives. See
[discopt_integration.md](discopt_integration.md) for a worked example.

## When to use which

| Need | Format |
|------|--------|
| Round-trip in Python | JSON spec |
| Inference in C++ / browser / mobile | ONNX |
| OMLT / Pyomo MILP embedding | ONNX |
| MILP / MINLP solve in pure Python (no Pyomo) | discopt adapter |
| Custom solver / spec for scipy / HiGHS | JSON spec → user-built MILP arrays |
