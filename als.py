"""Implicit feedback matrix factorisation (Hu, Koren and Volinsky, 2008).

Confidence ``c_ui = 1 + alpha * r_ui`` and preference ``p_ui = 1 if r_ui > 0``. Each half
step solves a regularised weighted least squares problem per row using the classic
``Y^T Y`` precomputation, which makes the cost proportional to the number of non zeros.

``fold_in`` computes a user vector on the fly from that user's latest interactions against
fixed item factors, so recommendations react to new activity immediately without waiting
for the next scheduled training run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(slots=True)
class ALSConfig:
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 15
    alpha: float = 20.0
    seed: int = 42


class ImplicitALS:
    def __init__(self, config: ALSConfig) -> None:
        self.config = config
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._gram: np.ndarray | None = None
        self._gram_source: int | None = None

    def item_gram(self) -> np.ndarray:
        """Return Y^T Y, computed once per set of item factors and reused by every fold in.

        Without this cache each request would multiply the full item factor matrix, which is
        O(items x factors^2) and dominates latency on large catalogues.
        """
        if self.item_factors is None:
            raise RuntimeError("Model is not fitted")
        if self._gram is None or self._gram_source != id(self.item_factors):
            self._gram = (self.item_factors.T @ self.item_factors).astype(np.float32)
            self._gram_source = id(self.item_factors)
        return self._gram

    def fit(self, interactions: sparse.csr_matrix) -> ImplicitALS:
        """Fit on a users by items matrix of non negative implicit feedback strengths."""
        cfg = self.config
        n_users, n_items = interactions.shape
        rng = np.random.default_rng(cfg.seed)
        scale = 0.01
        users = rng.normal(0, scale, (n_users, cfg.factors)).astype(np.float32)
        items = rng.normal(0, scale, (n_items, cfg.factors)).astype(np.float32)
        confidence = interactions.astype(np.float32).tocsr()
        confidence.data = cfg.alpha * confidence.data
        confidence_t = confidence.T.tocsr()
        for _ in range(cfg.iterations):
            users = self._solve(confidence, items, cfg.regularization)
            items = self._solve(confidence_t, users, cfg.regularization)
        self.user_factors = users
        self.item_factors = items
        return self

    @staticmethod
    def _solve(confidence: sparse.csr_matrix, fixed: np.ndarray, reg: float) -> np.ndarray:
        n_rows = confidence.shape[0]
        k = fixed.shape[1]
        yty = fixed.T @ fixed
        regularizer = reg * np.eye(k, dtype=np.float32)
        out = np.zeros((n_rows, k), dtype=np.float32)
        indptr, indices, data = confidence.indptr, confidence.indices, confidence.data
        for row in range(n_rows):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue
            cols = indices[start:end]
            conf = data[start:end]  # this is alpha * r, so c minus 1
            y = fixed[cols]
            a = yty + (y.T * conf) @ y + regularizer
            b = (y.T * (1.0 + conf)).sum(axis=1)
            out[row] = np.linalg.solve(a, b)
        return out

    def fold_in(self, item_indices: np.ndarray, strengths: np.ndarray) -> np.ndarray | None:
        """Solve for a user vector given interacted item indices and feedback strengths."""
        if self.item_factors is None or len(item_indices) == 0:
            return None
        cfg = self.config
        y_all = self.item_factors
        yty = self.item_gram()
        y = y_all[item_indices]
        conf = cfg.alpha * strengths.astype(np.float32)
        a = yty + (y.T * conf) @ y + cfg.regularization * np.eye(y_all.shape[1], dtype=np.float32)
        b = (y.T * (1.0 + conf)).sum(axis=1)
        return np.linalg.solve(a, b).astype(np.float32)

    def score_items(self, user_vector: np.ndarray, item_indices: np.ndarray | None = None) -> np.ndarray:
        if self.item_factors is None:
            raise RuntimeError("Model is not fitted")
        factors = self.item_factors if item_indices is None else self.item_factors[item_indices]
        return factors @ user_vector

    def similar_items(self, item_index: int, top_n: int = 50) -> list[tuple[int, float]]:
        if self.item_factors is None:
            raise RuntimeError("Model is not fitted")
        factors = self.item_factors
        norms = np.linalg.norm(factors, axis=1)
        norms[norms == 0] = 1.0
        target = factors[item_index] / norms[item_index]
        sims = (factors @ target) / norms
        sims[item_index] = -np.inf
        top = np.argpartition(-sims, min(top_n, len(sims) - 1))[:top_n]
        return sorted(((int(i), float(sims[i])) for i in top if np.isfinite(sims[i])), key=lambda x: -x[1])
