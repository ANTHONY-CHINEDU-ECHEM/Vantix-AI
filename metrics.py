"""Standard offline ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str] | dict[str, float], k: int) -> float:
    gains = relevant if isinstance(relevant, dict) else {r: 1.0 for r in relevant}
    dcg = sum(gains.get(r, 0.0) / math.log2(i + 2) for i, r in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def summarise(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: round(sum(q[key] for q in per_query) / len(per_query), 4) for key in keys}
