"""ActiveLearner: ask/tell primitive + run sugar with campaign persistence.

State machine:

  init       → no data yet; ask() raises.
  primed     → data observed; tree refit; ask() returns next batch.
  asked      → batch handed out; pending until tell() consumes it.

Persistence (campaign file):
- X_observed, y_observed
- jax PRNGKey state
- hyperparameters
- per-round acquisition history
- pending batches

Does NOT persist the trained tree (refit on load).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc


class _AcquisitionLike(Protocol):
    def score(self, model: Any, X: jnp.ndarray) -> np.ndarray: ...


class _BatcherLike(Protocol):
    def select(self, X: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray: ...


@dataclass
class _PendingBatch:
    batch_id: str
    X: np.ndarray  # (k, n_features)
    scores: np.ndarray  # (k,) acquisition scores at proposal time
    answered_indices: list[int] = field(default_factory=list)


@dataclass
class _Round:
    round_index: int
    batch_id: str
    X: np.ndarray
    y: np.ndarray
    scores: np.ndarray


class ActiveLearner:
    """Batch active learner over a fitted tree surrogate.

    Parameters
    ----------
    model_factory : callable returning a fresh, unfitted regressor.
    acquisition   : object with `.score(model, X) -> ndarray`. EI / UncertaintySampler / etc.
    batcher       : object with `.select(X, scores, k) -> indices`.
    bounds        : (n_features, 2) array of [low, high] per feature.
    batch_size    : number of points each ask() returns.
    seed          : initial PRNG seed.
    candidate_pool: optional fixed candidate set; otherwise Sobol-sampled.
    candidate_oversample : factor of batch_size when sampling Sobol.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        acquisition: _AcquisitionLike,
        batcher: _BatcherLike,
        bounds: np.ndarray,
        *,
        batch_size: int = 8,
        seed: int = 0,
        candidate_pool: Optional[np.ndarray] = None,
        candidate_oversample: int = 16,
    ) -> None:
        self.model_factory = model_factory
        self.acquisition = acquisition
        self.batcher = batcher
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_features, 2)")
        self.batch_size = int(batch_size)
        self.candidate_pool = (
            np.asarray(candidate_pool, dtype=np.float64)
            if candidate_pool is not None
            else None
        )
        self.candidate_oversample = int(candidate_oversample)

        # We persist `seed` and `rng_step` (an int counter) instead of the
        # raw `PRNGKey` array. Reconstruction is `key = fold_in(seed, step)`.
        # This survives JAX's typed-key-array migration without surprise.
        self._seed = int(seed)
        self._rng_step = 0
        self.X_observed = np.zeros((0, self.bounds.shape[0]), dtype=np.float64)
        self.y_observed = np.zeros((0,), dtype=np.float64)
        self.model: Optional[Any] = None
        self.pending: dict[str, _PendingBatch] = {}
        self.history: list[_Round] = []

    def _next_rng_key(self) -> jax.Array:
        """Deterministic per-step PRNG key derived from (seed, step)."""
        key = jax.random.fold_in(jax.random.PRNGKey(self._seed), self._rng_step)
        self._rng_step += 1
        return key

    # ---- core primitives ----

    def tell(
        self,
        batch_id_or_X: Union[str, np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        *,
        indices: Optional[list[int]] = None,
    ) -> None:
        """Ingest results.

        Two modes:
        - `tell(batch_id, y, indices=None)`: respond to a previously
          handed-out batch. `indices` lists which rows of the original
          batch were actually evaluated; defaults to all rows.
        - `tell(X, y)`: provide initial training data not from `ask()`.
        """
        y_raw = np.asarray(y, dtype=np.float64)
        if y_raw.ndim > 1 and y_raw.shape[-1] > 1:
            raise ValueError(
                "ActiveLearner currently supports only single-target objectives "
                f"(got y with shape {y_raw.shape}). Acquisitions like "
                "ExpectedImprovement assume a scalar best-y. Multi-target AL "
                "requires a scalarisation or multi-objective acquisition."
            )
        y_arr = y_raw.reshape(-1)

        if isinstance(batch_id_or_X, str):
            batch_id = batch_id_or_X
            if batch_id not in self.pending:
                raise KeyError(f"Unknown or already-consumed batch_id: {batch_id}")
            pending = self.pending[batch_id]
            if indices is None:
                if len(y_arr) != len(pending.X):
                    raise ValueError(
                        f"y has {len(y_arr)} entries but pending batch has {len(pending.X)} rows; "
                        f"pass indices=... for partial responses."
                    )
                indices = list(range(len(pending.X)))
            else:
                if len(y_arr) != len(indices):
                    raise ValueError("len(y) must match len(indices)")
                if any(i in pending.answered_indices for i in indices):
                    raise ValueError("Some indices in this batch were already answered")
            X_eval = pending.X[indices]
            self._extend_data(X_eval, y_arr)
            pending.answered_indices.extend(indices)
            self.history.append(
                _Round(
                    round_index=len(self.history),
                    batch_id=batch_id,
                    X=X_eval,
                    y=y_arr,
                    scores=pending.scores[indices],
                )
            )
            # If the batch is fully answered, drop it.
            if len(pending.answered_indices) == len(pending.X):
                del self.pending[batch_id]
        else:
            X_arr = np.asarray(batch_id_or_X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr[None, :]
            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("X and y must have matching first dim")
            self._extend_data(X_arr, y_arr)

        self._refit_model()

    def ask(self) -> tuple[str, np.ndarray]:
        """Propose the next batch. Returns (batch_id, X_batch)."""
        if self.model is None:
            raise RuntimeError("No data yet. Call tell(X_init, y_init) before ask().")

        # Pull pending unanswered points back into the candidate pool by
        # excluding them so we don't re-propose. (Simpler: never exclude,
        # since the acquisition already weights novelty via uncertainty.)
        candidates = self._sample_candidates()

        # Acquisition score
        if hasattr(self.acquisition, "update_best"):
            self.acquisition.update_best(self.y_observed)
        scores = np.asarray(self.acquisition.score(self.model, jnp.asarray(candidates)))

        idx = self.batcher.select(candidates, scores, self.batch_size)
        X_batch = candidates[idx]
        scores_batch = scores[idx]

        batch_id = uuid.uuid4().hex[:12]
        self.pending[batch_id] = _PendingBatch(
            batch_id=batch_id, X=X_batch, scores=scores_batch
        )
        return batch_id, X_batch

    def run(
        self,
        run_experiment: Callable[[np.ndarray], np.ndarray],
        n_rounds: int,
    ) -> None:
        """Sugar: alternate ask() and tell() with a user-provided callback.

        run_experiment is called as `run_experiment(X_batch) -> y_batch`.
        """
        for _ in range(n_rounds):
            batch_id, X_batch = self.ask()
            y_batch = np.asarray(run_experiment(X_batch), dtype=np.float64).reshape(-1)
            self.tell(batch_id, y_batch)

    # ---- persistence ----

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "version": 2,
            "X_observed": self.X_observed.tolist(),
            "y_observed": self.y_observed.tolist(),
            "seed": self._seed,
            "rng_step": self._rng_step,
            "bounds": self.bounds.tolist(),
            "batch_size": self.batch_size,
            "candidate_oversample": self.candidate_oversample,
            "candidate_pool": (
                self.candidate_pool.tolist() if self.candidate_pool is not None else None
            ),
            "pending": {
                bid: {
                    "batch_id": pb.batch_id,
                    "X": pb.X.tolist(),
                    "scores": pb.scores.tolist(),
                    "answered_indices": pb.answered_indices,
                }
                for bid, pb in self.pending.items()
            },
            "history": [
                {
                    "round_index": r.round_index,
                    "batch_id": r.batch_id,
                    "X": r.X.tolist(),
                    "y": r.y.tolist(),
                    "scores": r.scores.tolist(),
                }
                for r in self.history
            ],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        model_factory: Callable[[], Any],
        acquisition: _AcquisitionLike,
        batcher: _BatcherLike,
    ) -> "ActiveLearner":
        d = json.loads(Path(path).read_text())
        bounds = np.asarray(d["bounds"])
        cp = d.get("candidate_pool")
        # Backwards compat: v1 stored the raw PRNGKey under "rng_key";
        # v2 stores `seed` + `rng_step`. New saves always use v2.
        if "seed" in d:
            seed = int(d["seed"])
            step = int(d.get("rng_step", 0))
        else:
            # v1 fallback: re-seed from the first uint32 of the stored key.
            arr = np.asarray(d.get("rng_key", [0, 0]), dtype=np.uint32).reshape(-1)
            seed = int(arr[0]) if arr.size else 0
            step = 0
        loop = cls(
            model_factory=model_factory,
            acquisition=acquisition,
            batcher=batcher,
            bounds=bounds,
            batch_size=int(d["batch_size"]),
            seed=seed,
            candidate_pool=np.asarray(cp) if cp is not None else None,
            candidate_oversample=int(d.get("candidate_oversample", 16)),
        )
        loop._rng_step = step
        loop.X_observed = np.asarray(d["X_observed"], dtype=np.float64).reshape(
            -1, bounds.shape[0]
        )
        loop.y_observed = np.asarray(d["y_observed"], dtype=np.float64).reshape(-1)
        loop.pending = {
            bid: _PendingBatch(
                batch_id=p["batch_id"],
                X=np.asarray(p["X"], dtype=np.float64),
                scores=np.asarray(p["scores"], dtype=np.float64),
                answered_indices=list(p.get("answered_indices", [])),
            )
            for bid, p in d.get("pending", {}).items()
        }
        loop.history = [
            _Round(
                round_index=r["round_index"],
                batch_id=r["batch_id"],
                X=np.asarray(r["X"]),
                y=np.asarray(r["y"]),
                scores=np.asarray(r["scores"]),
            )
            for r in d.get("history", [])
        ]
        if loop.X_observed.shape[0] > 0:
            loop._refit_model()
        return loop

    # ---- internals ----

    def _extend_data(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_observed = np.concatenate([self.X_observed, np.asarray(X)], axis=0)
        self.y_observed = np.concatenate([self.y_observed, np.asarray(y)], axis=0)

    def _refit_model(self) -> None:
        self.model = self.model_factory()
        self.model.fit(jnp.asarray(self.X_observed), jnp.asarray(self.y_observed))

    def _sample_candidates(self) -> np.ndarray:
        if self.candidate_pool is not None:
            return self.candidate_pool
        n_target = self.batch_size * self.candidate_oversample
        # Sobol balance properties want a power of 2 sample count.
        n = 1 << max(1, int(np.ceil(np.log2(max(n_target, 2)))))
        d = self.bounds.shape[0]
        # Sobol-sample with a fresh JAX-seeded numpy RNG so save/load gives
        # reproducibility. The key is fold-in(seed, step) so callers
        # observe the same candidate sequence after save+load.
        sub = self._next_rng_key()
        seed = int(jax.random.randint(sub, (), 0, 2**31 - 1))
        sampler = qmc.Sobol(d=d, seed=seed)
        u = sampler.random(n)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + u * (hi - lo)
