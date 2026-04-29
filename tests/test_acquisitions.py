"""Direct tests for acquisition functions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_ldt import (
    LinearTreeRegressor,
    MaxVariance,
    ProbabilityOfImprovement,
    UncertaintySampler,
)


def test_probability_of_improvement_basic(rng) -> None:
    X = rng.uniform(-2.0, 2.0, size=(80, 1)).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    poi = ProbabilityOfImprovement(direction="min")
    poi.update_best(y)
    Xc = rng.uniform(-2.0, 2.0, size=(50, 1)).astype(np.float64)
    scores = np.asarray(poi.score(model, jnp.asarray(Xc)))
    assert scores.shape == (50,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_max_variance_returns_squared_sigma(rng) -> None:
    """`MaxVariance` returns σ² (matching its name); `UncertaintySampler`
    returns σ. They are argmax-equivalent on a fixed candidate set but
    their score values differ by a square."""
    from jax_ldt import LinearTreeRegressor

    X = rng.uniform(-2, 2, size=(80, 1)).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=10).fit(X, y)
    Xc = rng.uniform(-2, 2, size=(40, 1)).astype(np.float64)

    sigma = np.asarray(UncertaintySampler().score(model, jnp.asarray(Xc)))
    var = np.asarray(MaxVariance().score(model, jnp.asarray(Xc)))
    np.testing.assert_allclose(var, sigma ** 2, rtol=1e-6, atol=1e-12)
    # Argmax must agree (squaring is monotone on σ ≥ 0).
    assert int(np.argmax(sigma)) == int(np.argmax(var))
