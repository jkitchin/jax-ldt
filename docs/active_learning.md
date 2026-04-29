# Active learning

The `ActiveLearner` class drives a batch active-learning loop with the
trained tree as surrogate. Two ergonomics share one implementation:

- **`ask` / `tell`** — primitive for slow real-world experiments. Save
  the campaign to disk, run the lab work, come back tomorrow and `tell`
  it the results.
- **`run`** — sugar that calls `ask` / `tell` in a loop with a
  user-supplied callback. Best for in-silico experiments.

## Manual loop

```python
import numpy as np
from jax_ldt import (
    ActiveLearner, LinearTreeRegressor,
    ExpectedImprovement, GreedyMaxMinBatchSelector,
)

bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)

loop = ActiveLearner(
    model_factory=lambda: LinearTreeRegressor(max_depth=5, max_bins=8),
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    bounds=bounds,
    batch_size=8,
    seed=42,
)

# Initial random data
rng = np.random.default_rng(0)
X0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
y0 = (X0**2).sum(axis=1)
loop.tell(X0, y0)

# Lab session 1
batch_id, X_batch = loop.ask()
loop.save("campaign.json")

# (run experiments, possibly across days; only some succeed)
y_partial = ...               # results for indices 0, 2, 5
loop.tell(batch_id, y_partial, indices=[0, 2, 5])
loop.save("campaign.json")

# Lab session 2
loop = ActiveLearner.load(
    "campaign.json",
    model_factory=lambda: LinearTreeRegressor(max_depth=5, max_bins=8),
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
)
y_rest = ...                  # results for the missing rows
loop.tell(batch_id, y_rest, indices=[1, 3, 4, 6, 7])
```

## Sugar

The `run(...)` shortcut alternates `ask()` and `tell()` for you. It
still requires that the loop has been primed with at least one
`tell(...)` call so the surrogate has training data; otherwise the
first `ask()` raises `RuntimeError`.

```python
import numpy as np
from jax_ldt import (
    ActiveLearner, LinearTreeRegressor,
    ExpectedImprovement, GreedyMaxMinBatchSelector,
)

bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)
loop = ActiveLearner(
    model_factory=lambda: LinearTreeRegressor(max_depth=5, max_bins=8),
    acquisition=ExpectedImprovement(direction="min"),
    batcher=GreedyMaxMinBatchSelector(diversity_weight=0.4),
    bounds=bounds,
    batch_size=8,
    seed=42,
)

def f(X):
    return (X ** 2).sum(axis=1)        # in-silico oracle

# Prime with a small initial design before calling run().
rng = np.random.default_rng(0)
X_init = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
loop.tell(X_init, f(X_init))

loop.run(run_experiment=f, n_rounds=10)
print("best so far:", loop.y_observed.min())
```

## Acquisitions

| class | when to use |
|-------|-------------|
| `UncertaintySampler` | pure exploration; score = σ |
| `ExpectedImprovement(direction="min")` | classic Bayesian-optimisation style |
| `ProbabilityOfImprovement` | conservative variant of EI |
| `MaxVariance` | pure exploration; score = σ² (argmax-equivalent to `UncertaintySampler`) |

## Batchers

| class | strategy |
|-------|----------|
| `TopKBatchSelector` | naive top-k by score |
| `GreedyMaxMinBatchSelector` | score weighted by min-distance to chosen — recommended default |
| `DiverseBatchSelector` | KMeans-cluster top-N candidates and pick highest-scoring per cluster |

## Campaign persistence

`save(path)` and `load(path, model_factory=…, acquisition=…, batcher=…)`
write and read a JSON file with:

- `X_observed`, `y_observed`,
- `jax.random.PRNGKey` state,
- per-round acquisition history,
- pending batches handed out by `ask` but not fully `tell`-ed.

The trained tree itself is *not* persisted — `load` refits from the
data. This keeps the campaign file decoupled from internal `Tree` layout
(no version-coupling).

## Partial `tell` semantics

`ask()` returns a `(batch_id, X_batch)` tuple. `tell(batch_id, y,
indices=None)` accepts a subset of the original batch via `indices`.
Unanswered rows stay in the pending queue; the next `ask()` may
re-propose them or replace them based on the latest model. The
contract: no row of `X_batch` is ever silently dropped or duplicated.

## Tips

- For continuous spaces, the default Sobol-sampled candidate pool of
  `batch_size * 16` works well. Override `candidate_pool=…` for grid
  problems.
- To use `discopt.doe` for FIM-optimal batches, see
  [discopt_integration.md](discopt_integration.md).
