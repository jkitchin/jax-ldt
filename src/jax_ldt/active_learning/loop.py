"""ActiveLearner: ask/tell primitive + run sugar with campaign persistence.

State machine:

  init       → no data yet; ask() raises.
  primed     → data observed; tree refit; ask() returns next batch.
  asked      → batch handed out; pending until tell() consumes it.

Targets:
- `y` may be `(n,)` / `(n, 1)` (single target) or `(n, k)` (multi-target).
  Multi-target campaigns require an acquisition that advertises
  `requires_scalar_target = False` (UncertaintySampler, MaxVariance);
  pairing one with EI/PI raises. The target count is fixed by the first
  `tell()`.

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
    """Minimal acquisition interface.

    Optional attributes/methods the loop uses when present:

    * ``requires_scalar_target: bool`` — declares whether the acquisition
      needs a scalar objective. Missing means ``True`` (conservative).
    * ``update_best(y_observed)`` — called before scoring, for
      best-observed-y acquisitions.
    * ``update_target_scales(y_observed)`` — called before scoring, to
      supply fallback per-target scales for multi-target reductions.
    """

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


def _normalize_y(y: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    """Coerce user `y` to `(n,)` (single target) or `(n, k)` (multi).

    `(n, 1)` collapses to `(n,)` so single-target campaigns keep exactly
    their historical shape regardless of how the caller spelled it.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0] if arr.shape[1] == 1 else arr
    raise ValueError(
        f"y must be 1-D (n,) or 2-D (n, n_targets); got shape {arr.shape}."
    )


def _n_targets_of(y: np.ndarray) -> int:
    return 1 if y.ndim == 1 else int(y.shape[1])


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
        # Whether the batcher arrived with a user-set seed. Recorded ONCE
        # here: after the first ask() fills in a per-round seed, the field
        # is no longer None, so re-checking `is None` each round would
        # freeze it. If the user set a seed, we never override it.
        self._batcher_seed_user_set = getattr(batcher, "seed", None) is not None
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

    @property
    def n_targets(self) -> int:
        """Number of targets in the observed data (1 until y is 2-D)."""
        return _n_targets_of(self.y_observed)

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

        `y` is `(n,)` / `(n, 1)` for a single target or `(n, k)` for `k`
        targets; `(n, 1)` is normalised to `(n,)`. The first `tell()`
        fixes the campaign's target count.
        """
        y_arr = _normalize_y(y)
        # Multi-target data is only a problem for acquisitions that score
        # against a scalar best-y; the surrogate, the UQ path and the
        # exploration acquisitions all handle (n, k) natively. Check the
        # pairing, not the data.
        self._check_acquisition_targets(_n_targets_of(y_arr))

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
                n_rows = len(pending.X)
                if any((i < 0 or i >= n_rows) for i in indices):
                    raise ValueError(
                        f"indices must be in [0, {n_rows}); got {indices}"
                    )
                if len(set(indices)) != len(indices):
                    raise ValueError(f"duplicate indices in a single tell: {indices}")
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

        candidates = self._sample_candidates()

        # In fixed-pool mode a tree surrogate's leaf-constant uncertainty
        # does not collapse at observed points, so an already-labeled or
        # already-pending pool row can top-rank round after round. Exclude
        # them by exact row identity (pool rows are reused verbatim, so
        # float equality is exact) before scoring. Sobol mode draws fresh
        # points each ask(), so no exclusion is needed there.
        if self.candidate_pool is not None:
            used_rows = []
            if self.X_observed is not None and len(self.X_observed) > 0:
                used_rows.append(np.asarray(self.X_observed, dtype=np.float64))
            for pb in self.pending.values():
                used_rows.append(np.asarray(pb.X, dtype=np.float64))
            if used_rows:
                used_set = {tuple(r) for r in np.concatenate(used_rows, axis=0)}
                keep = np.array(
                    [tuple(r) not in used_set for r in candidates], dtype=bool
                )
                candidates = candidates[keep]
            if len(candidates) < self.batch_size:
                raise RuntimeError(
                    f"candidate pool exhausted: {len(candidates)} unlabeled/"
                    f"unpending points remain but batch_size={self.batch_size}. "
                    "Enlarge the pool or answer pending batches first."
                )

        # Acquisition score. Re-check the pairing here too: the
        # acquisition is a mutable attribute, so it may have been swapped
        # for a scalar-only one after multi-target data was ingested.
        self._check_acquisition_targets(self.n_targets)
        if hasattr(self.acquisition, "update_best"):
            self.acquisition.update_best(self.y_observed)
        # Fallback per-target scales for multi-target reductions. An
        # acquisition with explicit `target_scales` ignores these.
        if hasattr(self.acquisition, "update_target_scales"):
            self.acquisition.update_target_scales(self.y_observed)
        scores = np.asarray(self.acquisition.score(self.model, jnp.asarray(candidates)))

        # Give a seed-aware batcher (e.g. DiverseBatchSelector) that has NO
        # user-set seed a reproducible per-round seed derived from the
        # loop's fold-in RNG, so k-means++ seeding varies across rounds yet
        # survives save/load. If the user set an explicit seed, respect it.
        # We consume an RNG key either way so the candidate/seed stream
        # stays deterministic regardless of which branch runs.
        _round_seed = int(jax.random.randint(self._next_rng_key(), (), 0, 2**31 - 1))
        if hasattr(self.batcher, "seed") and not self._batcher_seed_user_set:
            self.batcher.seed = _round_seed
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
            y_batch = _normalize_y(run_experiment(X_batch))
            self.tell(batch_id, y_batch)

    # ---- persistence ----

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "version": 2,
            "X_observed": self.X_observed.tolist(),
            "y_observed": self.y_observed.tolist(),
            # Metadata: `load` infers the layout from `y_observed`'s
            # nesting, but a human or external reader of the campaign
            # file should not have to.
            "n_targets": self.n_targets,
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
        # `y_observed` is stored nested for multi-target campaigns and
        # flat for single-target ones, so np.asarray recovers either
        # shape. A campaign saved before its first tell() has no targets
        # fixed yet, so it restores to (0,) exactly as a fresh loop does.
        y_loaded = np.asarray(d["y_observed"], dtype=np.float64)
        loop.y_observed = y_loaded if y_loaded.size else np.zeros((0,), dtype=np.float64)
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

    def _check_acquisition_targets(self, n_targets: int) -> None:
        """Reject only scalar-only acquisition + multi-target data.

        The capability lives on the acquisition (`requires_scalar_target`);
        acquisitions that do not declare it are assumed scalar-only.
        """
        if n_targets <= 1:
            return
        if getattr(self.acquisition, "requires_scalar_target", True):
            raise ValueError(
                f"{type(self.acquisition).__name__} supports only single-target "
                f"objectives (got y with {n_targets} targets): it scores against "
                "a scalar best-y. For multi-target campaigns use an acquisition "
                "with requires_scalar_target=False (UncertaintySampler, "
                "MaxVariance), or pass a scalarisation of y."
            )

    def _extend_data(self, X: np.ndarray, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.float64)
        if self.y_observed.shape[0] == 0:
            # First data in: the target layout is fixed here.
            self.y_observed = np.zeros((0,) + y.shape[1:], dtype=np.float64)
        elif self.y_observed.shape[1:] != y.shape[1:]:
            raise ValueError(
                f"y has {_n_targets_of(y)} target(s) but this campaign already "
                f"holds {self.n_targets}; the target count is fixed at the first "
                "tell()."
            )
        self.X_observed = np.concatenate([self.X_observed, np.asarray(X)], axis=0)
        self.y_observed = np.concatenate([self.y_observed, y], axis=0)

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
