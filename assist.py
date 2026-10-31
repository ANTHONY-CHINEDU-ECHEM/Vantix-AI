"""Contextual assist: the hybrid of recommendation and RAG in a single call.

Designed for the task detail panel of a project management UI. Given the task a user is
looking at (and optionally a question), Vantix returns, in parallel:

* a grounded answer to the question, with citations
* knowledge items most relevant to the task, personalised to the user
* similar tasks (including completed ones, which often contain the solution)
* the people best placed to help
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from vantix.domain.models import Principal, Recommendation
from vantix.rag.service import RagAnswer, RagRequest, RagService
from vantix.recommender.engine import (
    RecommendationContext,
    RecommendationEngine,
    RecommendationKind,
    RecommendationQuery,
)


@dataclass(slots=True)
class AssistRequest:
    tenant_id: str
    principal: Principal
    item_id: str | None = None
    question: str | None = None
    project_id: str | None = None
    k: int = 5


@dataclass(slots=True)
class AssistResponse:
    answer: RagAnswer | None
    documents: list[Recommendation] = field(default_factory=list)
    similar_tasks: list[Recommendation] = field(default_factory=list)
    experts: list[Recommendation] = field(default_factory=list)


class AssistService:
    def __init__(self, recommender: RecommendationEngine, rag: RagService) -> None:
        self.recommender = recommender
        self.rag = rag
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="vantix_assist")

    def assist(self, req: AssistRequest) -> AssistResponse:
        context = RecommendationContext(item_id=req.item_id, query=req.question, project_id=req.project_id)

        def rec(kind: RecommendationKind, **extra: object) -> list[Recommendation]:
            return self.recommender.recommend(
                RecommendationQuery(req.tenant_id, req.principal, kind, k=req.k, context=context, **extra)  # type: ignore[arg-type]
            )

        from vantix.domain.models import ItemKind

        documents = self._pool.submit(rec, RecommendationKind.DOCUMENTS)
        similar = self._pool.submit(rec, RecommendationKind.SIMILAR, item_kinds=[ItemKind.TASK])
        experts = self._pool.submit(rec, RecommendationKind.EXPERTS)
        answer = None
        if req.question:
            answer = self._pool.submit(
                self.rag.answer,
                RagRequest(req.tenant_id, req.principal, req.question, context_item_id=req.item_id),
            ).result()
        return AssistResponse(
            answer=answer,
            documents=documents.result(),
            similar_tasks=similar.result(),
            experts=experts.result(),
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False)
