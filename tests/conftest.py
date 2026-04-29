"""Test fixtures and configuration."""

from __future__ import annotations

import os

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Force 64-bit JAX before any test module imports a `jax_ldt` symbol.

    pytest calls ``pytest_configure`` before collecting / importing test
    modules, so this is the earliest hook where we can flip the flag and
    still beat the user's own imports. We set both the env var (read
    once by JAX on first import) and the live config (so a JAX that has
    already been imported by a plugin still picks up x64).
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260427)


@pytest.fixture
def toy_1d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-3.0, 3.0, size=(200, 1))
    y = np.sin(X[:, 0]) + 0.05 * rng.standard_normal(200)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def branin_2d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform([-5.0, 0.0], [10.0, 15.0], size=(400, 2))
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    x1, x2 = X[:, 0], X[:, 1]
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def friedman1_6d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(0.0, 1.0, size=(500, 6))
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )
    return X.astype(np.float64), y.astype(np.float64)
