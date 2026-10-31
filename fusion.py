"""Rank fusion and diversification primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def weighted_rrf(
    rankings: dict[str, Sequence[str]], weights: dict[str, float], k: int = 60
) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion.

    ``score(d) = sum_r w_r / (k + rank_r(d))`` with ranks starting at 1. RRF is robust to the
    incomparable score scales of BM25 and cosine similarity, which is why it is the default.
    """
    fused: dict[str, float] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        if weight == 0:
            continue
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return fused


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def mmr(
    candidates: Sequence[T],
    relevance: Callable[[T], float],
    similarity: Callable[[T, T], float],
    k: int,
    lambda_: float = 0.75,
) -> list[T]:
    """Maximal Marginal Relevance selection balancing relevance against redundancy."""
    remaining = list(candidates)
    selected: list[T] = []
    while remaining and len(selected) < k:
        best_index = 0
        best_score = float("-inf")
        for i, candidate in enumerate(remaining):
            redundancy = max((similarity(candidate, s) for s in selected), default=0.0)
            score = lambda_ * relevance(candidate) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_index = i
        selected.append(remaining.pop(best_index))
    return selected


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
