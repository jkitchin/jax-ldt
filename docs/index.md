# jax-ldt

A pure-JAX rewrite of LLNL's
[hyperplanetree](https://github.com/LLNL/systems2atoms/tree/main/systems2atoms/hyperplanetree)
library, with first-class uncertainty quantification, an active-learning
loop, and a Pyomo-free path to MIP embedding.

What's here:

- **`LinearTreeRegressor`** — axis-aligned Linear Model Decision Trees.
- **`HyperplaneTreeRegressor`** — oblique-split (linear-combination)
  decision trees.
- **Uncertainty quantification** — linear propagation, quadratic
  difference, and Mondrian leaf conformal prediction intervals.
- **Active learning** — `ask`/`tell` primitives plus a `run` callback
  loop, with a campaign-file format that survives a crashed lab session.
- **Export** — solver-neutral JSON, hand-built ONNX (consumable by OMLT
  externally), and a `discopt` adapter for direct MINLP embedding.

Read the [quickstart](quickstart.md) to fit a model in five lines, then
the [algorithm notes](algorithm.md), [uncertainty guide](uncertainty.md),
[active learning guide](active_learning.md), and the
[MIP / surrogate-embedding recipes](mip_export.md).
