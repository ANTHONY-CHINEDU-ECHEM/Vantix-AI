"""Offline evaluation for retrieval and recommendation quality.

Retrieval: a JSON Lines file of labelled queries, one object per line:
``{"query": "...", "relevant_item_ids": ["..."], "principal": {"user_id": "...", ...}}``.

Recommendation: a strict temporal split. Interactions before the cutoff (by default the
80th percentile timestamp) are copied into an isolated in memory store, a model is trained
on them, and each user's later interactions with items they had not seen before become the
ground truth. Nothing after the cutoff can leak into training.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import insert

from vantix.config import LexicalBackend, VectorBackend
from vantix.domain.models import Interaction, InteractionType, ItemKind, Principal
from vantix.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)
from vantix.index.base import SearchFilter
from vantix.recommender.engine import RecommendationKind, RecommendationQuery
from vantix.retrieval.hybrid import RetrievalRequest
from vantix.storage import schema

if TYPE_CHECKING:  # pragma: no cover
    from vantix.container import Container


def evaluate_retrieval(container: Container, tenant_id: str, path: Path, k: int = 10) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    per_query: list[dict[str, float]] = []
    for row in rows:
        principal = Principal.model_validate(
            row.get("principal") or {"user_id": "evaluator", "is_admin": True}
        )
        relevant = set(row["relevant_item_ids"])
        hits = container.retriever.retrieve(
            RetrievalRequest(
                tenant_id, row["query"], SearchFilter(principal=principal), k=k * 3, diversify=False
            )
        )
        ranked = list(dict.fromkeys(h.chunk.item_id for h in hits))[:k]
        per_query.append(
            {
                f"precision@{k}": precision_at_k(ranked, relevant, k),
                f"recall@{k}": recall_at_k(ranked, relevant, k),
                f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
                "mrr": reciprocal_rank(ranked, relevant),
                f"hit_rate@{k}": hit_rate_at_k(ranked, relevant, k),
            }
        )
    return {"queries": len(rows), "metrics": summarise(per_query)}


def evaluate_recommender(
    source: Container, tenant_id: str, k: int = 10, cutoff_quantile: float = 0.8, min_history: int = 2
) -> dict[str, Any]:
    from vantix.container import Container

    events = list(source.repository.iter_interactions(tenant_id))
    if len(events) < 10:
        return {"error": "Not enough interactions for a temporal evaluation", "interactions": len(events)}
    times = sorted(e[4] for e in events)
    cutoff: datetime = times[int(len(times) * cutoff_quantile)]

    with tempfile.TemporaryDirectory() as tmp:
        settings = source.settings.model_copy(
            update={
                "database_url": "sqlite://",
                "vector_backend": VectorBackend.MEMORY,
                "lexical_backend": LexicalBackend.MEMORY,
                "model_dir": Path(tmp),
                "retrieval_cache_ttl_seconds": 0,
                "rec_training_interval_seconds": 0,
            }
        )
        sandbox = Container.build(settings)
        _copy_content(source, sandbox, tenant_id)
        sandbox.ingestion.rebuild_indexes(tenant_id)

        history: dict[str, set[str]] = defaultdict(set)
        train: list[Interaction] = []
        future: dict[str, set[str]] = defaultdict(set)
        for n, (user, item, itype, weight, occurred) in enumerate(events):
            if occurred < cutoff:
                train.append(
                    Interaction(
                        event_id=f"e{n}",
                        user_id=user,
                        item_id=item,
                        type=InteractionType(itype),
                        occurred_at=occurred,
                        value=weight,
                    )
                )
                if weight > 0:
                    history[user].add(item)
            elif weight > 0:
                future[user].add(item)
        sandbox.repository.add_interactions(tenant_id, train)
        sandbox.registry.put(sandbox.trainer.train(tenant_id, now=cutoff), persist=False)

        per_user: list[dict[str, float]] = []
        recommended: set[str] = set()
        for user, items in future.items():
            truth = items - history[user]
            if len(history[user]) < min_history or not truth:
                continue
            recs = sandbox.recommender.recommend(
                RecommendationQuery(
                    tenant_id=tenant_id,
                    principal=Principal(user_id=user, is_admin=True),
                    kind=RecommendationKind.DOCUMENTS,
                    k=k,
                    item_kinds=list(ItemKind),
                    diversify=False,
                )
            )
            ranked = [r.target_id for r in recs]
            recommended.update(ranked)
            per_user.append(
                {
                    f"precision@{k}": precision_at_k(ranked, truth, k),
                    f"recall@{k}": recall_at_k(ranked, truth, k),
                    f"ndcg@{k}": ndcg_at_k(ranked, truth, k),
                    f"hit_rate@{k}": hit_rate_at_k(ranked, truth, k),
                    "mrr": reciprocal_rank(ranked, truth),
                }
            )
        catalogue = sum(1 for _ in sandbox.repository.iter_items(tenant_id))
        sandbox.stop()
    metrics = summarise(per_user)
    metrics["catalogue_coverage"] = round(len(recommended) / catalogue, 4) if catalogue else 0.0
    return {"cutoff": cutoff.isoformat(), "users_evaluated": len(per_user), "metrics": metrics}


def _copy_content(source: Container, target: Container, tenant_id: str) -> None:
    """Copy items and chunks with their stored embeddings, so evaluation never re embeds."""
    from vantix.storage.repository import _item_to_row

    with target.repository.engine.begin() as conn:
        for stored in source.repository.iter_items(tenant_id):
            row = _item_to_row(tenant_id, stored.item, stored.content_hash, stored.embedding)
            conn.execute(insert(schema.items).values(**row))
        batch: list[dict[str, Any]] = []
        for chunk_row in source.repository.iter_chunk_rows(tenant_id):
            batch.append(chunk_row)
            if len(batch) >= 1000:
                conn.execute(insert(schema.chunks), batch)
                batch = []
        if batch:
            conn.execute(insert(schema.chunks), batch)
