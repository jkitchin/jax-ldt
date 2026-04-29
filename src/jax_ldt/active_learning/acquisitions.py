"""Acquisition functions: score candidate points for the next batch.

Every acquisition implements `.score(model, X_candidates) -> jnp.ndarray`,
returning a 1-D score (higher is better — we always pick max-score).

Acquisitions know about uncertainty calibration; the loop wires them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.uncertainty.linprop import linprop_uncertainty


class _ModelLike(Protocol):
    tree_: Optional[Tree]

    def predict(self, X: jnp.ndarray) -> jnp.ndarray: ...


def _sigma(model: _ModelLike, X: jnp.ndarray) -> jnp.ndarray:
    """Per-sample predictive σ. Always 1-D ``(N,)``.

    ``linprop_uncertainty`` returns ``(N, T)`` for multi-target trees;
    every acquisition in this module is single-output (EI / PI / UCB
    against a scalar best-y), so we reject multi-target inputs here
    rather than letting `(N, T)` flatten into a length-`N·T` array that
    silently misaligns with ``mu``.

    Returns a JAX array (kept on-device) so downstream EI/PI scoring
    runs without a numpy round-trip and stays vmap-friendly.
    """
    out = jnp.asarray(linprop_uncertainty(model.tree_, X))
    if out.ndim == 2:
        if out.shape[1] != 1:
            # n_targets is a static Python int on Tree, so this branch
            # decision is JIT-safe.
            raise ValueError(
                f"acquisition functions are single-output, but model.tree_ "
                f"has n_targets={out.shape[1]}. Reduce to a scalar target "
                f"or use a multi-target acquisition (not implemented)."
            )
        out = out[:, 0]
    return out


@dataclass
class UncertaintySampler:
    """Maximum-uncertainty acquisition.

    Score = σ_pred(x). Pure exploration.
    """

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        return np.asarray(_sigma(model, X_candidates))


@dataclass
class MaxVariance:
    """Maximum predictive-variance acquisition.

    Score = σ_pred(x)². Argmax-equivalent to :class:`UncertaintySampler`
    (squaring is monotone on σ ≥ 0), but the *score values* are σ² —
    matching the class name. Use this when downstream code consumes the
    raw score (e.g., portfolio acquisitions or adaptive temperature
    schedules) and expects variance units.
    """

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        sigma = _sigma(model, X_candidates)
        return np.asarray(sigma * sigma)


@dataclass
class ExpectedImprovement:
    """Gaussian expected improvement.

    Treats σ_pred as a Gaussian std around the mean prediction. Score is
    EI relative to the best observed `y` so far.

    direction: "min" → minimize y; "max" → maximize y. Default min.
    xi: exploration parameter (added to the gap). Default 0.0.
    """

    direction: str = "min"
    xi: float = 0.0
    _y_best: Optional[float] = None

    def update_best(self, y_observed: np.ndarray) -> None:
        y = np.asarray(y_observed).reshape(-1)
        if y.size == 0:
            return
        self._y_best = float(np.min(y) if self.direction == "min" else np.max(y))

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        if self._y_best is None:
            raise RuntimeError("Call update_best(y) before scoring.")
        mu = jnp.asarray(model.predict(X_candidates)).reshape(-1)
        sigma = jnp.maximum(_sigma(model, X_candidates).reshape(-1), 1e-9)

        if self.direction == "min":
            improvement = self._y_best - mu - self.xi
        else:
            improvement = mu - self._y_best - self.xi
        z = improvement / sigma
        ei = improvement * jstats.norm.cdf(z) + sigma * jstats.norm.pdf(z)
        return np.asarray(ei)


@dataclass
class ProbabilityOfImprovement:
    """Gaussian probability of improvement against best-observed."""

    direction: str = "min"
    xi: float = 0.0
    _y_best: Optional[float] = None

    def update_best(self, y_observed: np.ndarray) -> None:
        y = np.asarray(y_observed).reshape(-1)
        if y.size == 0:
            return
        self._y_best = float(np.min(y) if self.direction == "min" else np.max(y))

    def score(self, model: _ModelLike, X_candidates: jnp.ndarray) -> np.ndarray:
        if self._y_best is None:
            raise RuntimeError("Call update_best(y) before scoring.")
        mu = jnp.asarray(model.predict(X_candidates)).reshape(-1)
        sigma = jnp.maximum(_sigma(model, X_candidates).reshape(-1), 1e-9)
        if self.direction == "min":
            z = (self._y_best - mu - self.xi) / sigma
        else:
            z = (mu - self._y_best - self.xi) / sigma
        return np.asarray(jstats.norm.cdf(z))
