"""Batch selectors: turn (candidates, scores) into a chosen batch.

Each selector implements `.select(X_candidates, scores, k) -> indices`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopKBatchSelector:
    """Pick the k highest-scoring candidates. Naive; no diversity."""

    def select(self, X_candidates: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(-scores)
        return order[:k]


@dataclass
class GreedyMaxMinBatchSelector:
    """Greedy with min-distance penalty.

    At each step, pick the candidate maximizing
        score(x) - lambda * (max_distance_to_already_chosen)^(-1)
    using the simple form
        adjusted_score(x) = score(x) - lambda * d_min(x, already_chosen) ^ (-1)
    Equivalently, we just take the highest-score point that is at least
    `min_distance_factor * max_pairwise_distance` away from all chosen.

    For practical use we implement the standard "scoring with farthest-
    point penalty": for each candidate compute its current min distance
    to the chosen set; weight the score by tanh(distance / scale).
    """

    diversity_weight: float = 0.5

    def select(self, X_candidates: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        X = np.asarray(X_candidates)
        n = X.shape[0]
        if k >= n:
            return np.arange(n)

        # Normalise scores to [0, 1] so the diversity term is comparable
        s = scores - scores.min()
        denom = s.max()
        if denom > 0:
            s = s / denom

        chosen: list[int] = []
        # Start with highest score
        chosen.append(int(np.argmax(s)))

        # Pre-compute pairwise distance feature scale
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges = np.where(ranges < 1e-8, 1.0, ranges)
        Xn = X / ranges  # normalised

        # min-dist to chosen, updated incrementally
        dist_to_chosen = np.linalg.norm(Xn - Xn[chosen[0]], axis=1)

        for _ in range(k - 1):
            # diversity term in [0, 1]: 1 if far, 0 if close
            divs = np.tanh(dist_to_chosen)
            adjusted = (1.0 - self.diversity_weight) * s + self.diversity_weight * divs
            adjusted[chosen] = -np.inf
            nxt = int(np.argmax(adjusted))
            chosen.append(nxt)
            new_dists = np.linalg.norm(Xn - Xn[nxt], axis=1)
            dist_to_chosen = np.minimum(dist_to_chosen, new_dists)

        return np.array(chosen, dtype=np.int64)


@dataclass
class DiverseBatchSelector:
    """KMeans-cluster the top-N scoring candidates and pick one per cluster.

    Cheap and effective: n_clusters = k. Each cluster contributes its
    highest-scoring candidate.
    """

    pool_factor: int = 5  # consider pool_factor * k top candidates
    diversity_weight: float = 0.3  # unused here; kept for API symmetry

    def select(self, X_candidates: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        X = np.asarray(X_candidates)
        n = X.shape[0]
        if k >= n:
            return np.arange(n)
        pool_size = min(n, self.pool_factor * k)
        order = np.argsort(-scores)
        pool = order[:pool_size]
        Xp = X[pool]

        # Simple k-means++-style seeding then 5 Lloyd iterations.
        rng = np.random.default_rng(0)
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges = np.where(ranges < 1e-8, 1.0, ranges)
        Xn = Xp / ranges

        # k-means++ seeding. Indices are local to ``Xp`` / ``Xn``.
        # Already-chosen centers are explicitly excluded from each draw
        # so duplicates can't be sampled (the standard ``+ 1e-12`` smoothing
        # would otherwise let any point — centers included — be re-drawn
        # if all squared distances collapse to zero).
        first = int(np.argmax(scores[pool]))
        centers = [first]
        d2 = np.sum((Xn - Xn[first]) ** 2, axis=1)
        for _ in range(k - 1):
            available = d2.copy()
            available[centers] = 0.0
            total = available.sum()
            if total > 0.0:
                probs = available / total
                nxt = int(rng.choice(len(Xn), p=probs))
            else:
                # All non-center points coincide with a center (duplicates
                # in Xn): fall back to picking any unused index.
                remaining = [i for i in range(len(Xn)) if i not in set(centers)]
                if not remaining:
                    break
                nxt = int(rng.choice(remaining))
            centers.append(nxt)
            d2 = np.minimum(d2, np.sum((Xn - Xn[nxt]) ** 2, axis=1))
        center_pts = Xn[centers]
        n_centers = len(centers)

        # 5 Lloyd iterations
        for _ in range(5):
            d = np.linalg.norm(Xn[:, None, :] - center_pts[None, :, :], axis=2)
            assigns = np.argmin(d, axis=1)
            for c in range(n_centers):
                if np.any(assigns == c):
                    center_pts[c] = Xn[assigns == c].mean(axis=0)

        # Within each cluster, pick highest-scoring candidate
        d = np.linalg.norm(Xn[:, None, :] - center_pts[None, :, :], axis=2)
        assigns = np.argmin(d, axis=1)
        chosen: list[int] = []
        used: set[int] = set()
        pool_scores = scores[pool]
        for c in range(n_centers):
            members = np.where(assigns == c)[0]
            if len(members) == 0:
                # No member: pick highest-score candidate not used yet
                for idx in np.argsort(-pool_scores):
                    if int(pool[idx]) not in used:
                        chosen.append(int(pool[idx]))
                        used.add(int(pool[idx]))
                        break
                continue
            best_local = members[np.argmax(pool_scores[members])]
            chosen.append(int(pool[best_local]))
            used.add(int(pool[best_local]))

        # If we ended k-means++ early (every pool point coincided with
        # a chosen center), pad with the highest-score unused candidates.
        if len(chosen) < k:
            for idx in np.argsort(-pool_scores):
                if int(pool[idx]) not in used:
                    chosen.append(int(pool[idx]))
                    used.add(int(pool[idx]))
                    if len(chosen) == k:
                        break
        return np.array(chosen, dtype=np.int64)
