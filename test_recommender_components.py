import numpy as np
from scipy import sparse

from vantix.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from vantix.recommender.als import ALSConfig, ImplicitALS
from vantix.recommender.graph import GraphBuilder


def test_als_learns_block_structure_and_fold_in() -> None:
    # Users 0 to 3 like items 0 to 3; users 4 to 7 like items 4 to 7.
    rows, cols = [], []
    for u in range(8):
        block = range(0, 4) if u < 4 else range(4, 8)
        for i in block:
            if (u + i) % 4 != 0:  # leave one item per user unseen
                rows.append(u)
                cols.append(i)
    m = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(8, 8))
    als = ImplicitALS(ALSConfig(factors=2, iterations=20, alpha=10, regularization=0.1)).fit(m)
    scores = als.score_items(als.user_factors[0])  # type: ignore[index]
    assert scores[:4].mean() > scores[4:].mean()
    folded = als.fold_in(np.array([4, 5]), np.array([1.0, 1.0]))
    assert folded is not None
    assert als.score_items(folded)[4:].mean() > als.score_items(folded)[:4].mean()
    similar = [i for i, _ in als.similar_items(0, 3)]
    assert all(i < 4 for i in similar)


def test_personalized_pagerank_prefers_local_structure() -> None:
    g = GraphBuilder()
    g.add("u:a", "i:1", 3)
    g.add("i:1", "p:x", 1)
    g.add("i:2", "p:x", 1)
    g.add("u:b", "i:3", 3)
    g.add("i:3", "p:y", 1)
    graph = g.build()
    scores = graph.personalized_pagerank({"u:a": 1.0})
    assert scores.get("i:2", 0) > scores.get("i:3", 0)
    assert graph.personalized_pagerank({"u:missing": 1.0}) == {}
    assert ("p:x", 1.0) in graph.neighbours("i:1")


def test_ranking_metrics() -> None:
    ranked = ["a", "b", "c"]
    assert precision_at_k(ranked, {"b"}, 2) == 0.5
    assert recall_at_k(ranked, {"b", "z"}, 3) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == 1 / 3
    assert hit_rate_at_k(ranked, {"z"}, 3) == 0.0
    assert ndcg_at_k(["a"], {"a"}, 1) == 1.0
