# Uncertainty quantification

Three composable methods, each with its own use case.

## Linear-propagation UQ

Closed-form per-sample uncertainty from leaf statistics saved at fit
time. For a leaf with `n` training points, sample mean `μ`, sample
variance `σ²`, and leaf MSE `m`:

    σ_pred(x) = sqrt(m/n + Σⱼ (xⱼ - μⱼ)² / ((n-1) σⱼ²))

The first term is the leaf's residual variance scaled by sample count
(epistemic); the second grows with distance from the leaf's training
mean (extrapolation penalty). Cheap, no calibration step required.

```python
from jax_ldt import LinearPropagationUQ
sigma = LinearPropagationUQ().predict(model.tree_, X)
```

## Quadratic-difference UQ

Refits each leaf with quadratic features (cross-products) and reports
`|linear_pred - quadratic_pred|` as a model-form sensitivity. Useful for
identifying regions where the linear assumption breaks down.

```python
from jax_ldt import QuadraticUQ

quq = QuadraticUQ(ridge=1e-5).calibrate(model.tree_, X_train, y_train)
sigma_q = quq.predict(model.tree_, X_new)
```

## Mondrian leaf conformal

Distribution-free prediction intervals with valid finite-sample
coverage. After fit, route a calibration set through the tree; the
per-leaf empirical `(1-α)` quantile of `|y - ŷ|` is the half-width:

```python
from jax_ldt import ConformalCalibrator

calib = ConformalCalibrator(alpha=0.1, mondrian=True).calibrate(
    model, X_cal, y_cal
)
lo, hi = calib.predict_interval(X_test, model=model)
```

Settings:

- `alpha`: miscoverage (e.g. 0.1 → 90% intervals).
- `mondrian=True`: per-leaf quantile (heteroscedasticity-aware).
- `mondrian=False`: a single global quantile across all leaves.
- `min_calibration_per_leaf`: leaves with fewer calibration points
  trigger the sparse-leaf fallback (see below).
- `sparse_leaf_strategy="global"` (default) or `"skip"`.

### Coverage caveats

The conformal guarantee is **marginal**: averaged over the calibration
draw, expected coverage on a fresh test point is `1 - α`. On any single
finite calibration set the empirical coverage can fluctuate by roughly
`±√(α(1-α)/N_test)`. With `N_test = 100` and `α = 0.1`, that's about
±3 percentage points; small calibration runs can easily land in the
85–93% band on a 90% nominal target.

Mondrian (per-leaf) conformal additionally preserves *conditional*
coverage on each leaf — *as long as* the leaf has enough calibration
data. **Sparse leaves break this**:

- `sparse_leaf_strategy="global"` (default): sparse leaves borrow the
  global residual quantile. This keeps marginal coverage close to
  nominal, but per-leaf coverage on the sparse buckets is no longer
  exchangeable; it can over- or under-cover depending on how their
  noise compares to the rest of the tree.
- `sparse_leaf_strategy="skip"`: sparse leaves get `NaN` intervals at
  predict time. Use this when valid per-leaf coverage matters more
  than always producing a finite interval.

A `UserWarning` lists the sparse leaves whenever calibration runs.

For the common case of "split the training data 80/20, calibrate on the
held-out 20":

```python
from jax_ldt import LinearTreeRegressor, fit_with_conformal

model, calib = fit_with_conformal(
    LinearTreeRegressor(max_depth=5),
    X, y, calibration_size=0.2, alpha=0.1,
)
```

## Calibration metrics

If you have `uncertainty-toolbox` installed, `calibration_metrics()`
returns accuracy, sharpness, calibration, and scoring-rule metrics in
one call:

```python
from jax_ldt.uncertainty import calibration_metrics

metrics = calibration_metrics(y_pred, sigma, y_true)
print(metrics["calibration"])
```
