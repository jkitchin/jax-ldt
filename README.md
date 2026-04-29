# jax-ldt

Pure-JAX linear and hyperplane decision trees for piecewise-linear
surrogate modeling, with first-class uncertainty quantification and a
batched active-learning loop.

This is a JAX rewrite of the PyTorch
[hyperplanetree](https://github.com/LLNL/systems2atoms/tree/main/systems2atoms/hyperplanetree)
library, omitting the Pyomo/OMLT runtime dependency. Trained models
export to a solver-neutral JSON spec and to ONNX, which can still be
consumed by OMLT externally for MILP embedding.

## Install

```bash
pip install -e ".[dev,onnx]"
```

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

See [docs/](docs/) for the full guide: algorithm details, uncertainty
quantification, active-learning loop, and MIP export.

## License

MIT
