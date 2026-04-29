# Benchmarks

Performance numbers are environment-dependent; the table below is
meant as an order-of-magnitude reference, not a hard claim.

| Operation | jax-ldt | upstream PyTorch | Notes |
|-----------|---------|------------------|-------|
| Fit LMDT, Branin 2D, 400 pts | ~0.6 s | ~0.4 s | Cold-start JIT cost dominates |
| Predict 1000 pts | ~50 ms | ~30 ms | After JIT; vmapped routing |
| Fit HT, Branin 2D | ~1.5 s | ~1.2 s | Hyperplane lift adds features |
| ONNX export | ~10 ms | n/a | hand-built ONNX graph |

The JIT compile happens on the first call to the inner split kernel;
subsequent splits with the same shape signature are fast.

## Reproducing

The benchmark suite (skipped in regular CI) is `tests/test_performance.py`
(planned). Run with:

```bash
pytest tests/test_performance.py --benchmark-only
```

## Where speed comes from

- **Inner kernel JIT.** `_evaluate_splits_kernel` is JIT-compiled, so
  each candidate-split evaluation is one compiled kernel call rather
  than a chain of Python ops.
- **Vmapped ridge fits.** Per-candidate ridge regressions are batched
  via `jax.vmap`, fusing K·B fits into one call.
- **Mask-based subdomain selection.** No Python-level slicing /
  reallocation between splits — the kernel sees the full (N, n_aug)
  tensor and a (N,) mask.

## Where it slows down

- **Tree growth Python loop.** Topology is data-dependent and not
  JIT-compatible. Each split call re-enters Python.
- **First split has JIT compile cost** (~0.5 s on CPU). Re-fitting trees
  with the same hyperparameters reuses the cached compilation.
