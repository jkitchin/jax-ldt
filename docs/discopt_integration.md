# discopt integration

[discopt](https://github.com/jkitchin/discopt) is a hybrid MINLP solver
with a JAX backend. jax-ldt provides two integration points.

## Embed a tree as MINLP constraints

```python
import discopt
import numpy as np

from jax_ldt import HyperplaneTreeRegressor
from jax_ldt.export import embed_in_discopt_model

# Train a surrogate
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(300, 2))
y = (X ** 2).sum(axis=1) + 0.05 * X[:, 0] * X[:, 1]
model = HyperplaneTreeRegressor(max_depth=4, max_bins=6, min_samples_leaf=15)
model.fit(X, y)

# Build the MINLP
m = discopt.Model("min-y")
x = m.continuous("x", shape=(2,), lb=-2, ub=2)
y_expr = embed_in_discopt_model(model, m, x, big_m=20.0)
m.minimize(y_expr[0])
result = m.solve()
print("argmin:", result.x, "min:", result.objective)
```

The `embed_in_discopt_model` helper creates lifted-feature variables,
binary leaf indicators, big-M routing constraints, and bilinear
auxiliary variables for the leaf regressions — yielding a MILP that any
discopt-compatible backend can solve.

For pure axis-aligned LMDTs with constant leaves, a lighter-weight
adapter is also available:

```python
from jax_ldt.export import to_discopt_decision_tree

dt = to_discopt_decision_tree(model)   # raises if model has linear leaves
# Use with discopt.nn.tree.TreeFormulation (see discopt docs).
```

## Use discopt for batch DoE

If you want Fisher-information-optimal experiment batches instead of
acquisition-function batches, plug `discopt.doe.batch_optimal_experiment`
into your active-learning loop. (This is documented as advanced; FIM
methods assume the surrogate is parametric, which the piecewise-linear
tree is only at the leaf level.)

A worked example is planned; for now see
[discopt's DoE tutorial](https://github.com/jkitchin/discopt/tree/main/docs)
and adapt by treating each leaf's coefficients as parameters.
