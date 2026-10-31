"""Retrieval augmented generation with personalisation, citations and abstention.

Flow:

1. Rewrite the question with recent conversation turns so follow ups retrieve correctly.
2. Retrieve with the hybrid retriever, filtered by the caller's permissions and personalised
   with graph affinity from the recommender.
3. Abstain early when nothing sufficiently relevant is found, rather than letting the model guess.
4. Pack the best passages into a token budget with numbered source headers.
5. Generate, then validate every citation against the sources actually supplied.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from vantix.config import Settings
from vantix.domain.models import Citation, ItemKind, Principal, ScoredChunk
from vantix.index.base import SearchFilter
from vantix.observability.metrics import RAG_ABSTENTIONS, Timer
from vantix.rag.llm import LLM, ChatMessage
from vantix.rag.prompts import ABSTAIN_MESSAGE, SYSTEM_PROMPT, USER_TEMPLATE
from vantix.recommender.engine import RecommendationEngine
from vantix.retrieval.hybrid import HybridRetriever, RetrievalRequest
from vantix.storage.repository import Repository
from vantix.text import count_tokens

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass(slots=True)
class RagRequest:
    tenant_id: str
    principal: Principal
    question: str
    history: list[ChatMessage] = field(default_factory=list)
    project_ids: list[str] | None = None
    item_kinds: list[ItemKind] | None = None
    context_item_id: str | None = None
    k: int = 8
    personalize: bool = True


@dataclass(slots=True)
class RagAnswer:
    answer: str
    citations: list[Citation]
    grounded: bool
    abstained: bool
    model: str
    usage: dict[str, int]
    retrieved: int


@dataclass(slots=True)
class PreparedContext:
    prompt: str
    sources: list[Citation]
    abstain: bool


class RagService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        repository: Repository,
        llm: LLM,
        recommender: RecommendationEngine,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.repo = repository
        self.llm = llm
        self.recommender = recommender

    def answer(self, req: RagRequest) -> RagAnswer:
        prepared, retrieved = self.prepare(req)
        if prepared.abstain:
            RAG_ABSTENTIONS.inc()
            return RagAnswer(ABSTAIN_MESSAGE, [], False, True, self.llm.name, {}, retrieved)
        with Timer("generate"):
            result = self.llm.generate(
                SYSTEM_PROMPT,
                [*req.history[-6:], ChatMessage("user", prepared.prompt)],
                self.settings.llm_max_output_tokens,
                self.settings.llm_temperature,
            )
        text, used = validate_citations(result.text, prepared.sources)
        return RagAnswer(
            answer=text,
            citations=used,
            grounded=bool(used),
            abstained=False,
            model=result.model,
            usage=result.usage,
            retrieved=retrieved,
        )

    def stream(self, req: RagRequest) -> tuple[list[Citation], Iterator[str], bool]:
        prepared, _ = self.prepare(req)
        if prepared.abstain:
            RAG_ABSTENTIONS.inc()
            return [], iter([ABSTAIN_MESSAGE]), True
        tokens = self.llm.stream(
            SYSTEM_PROMPT,
            [*req.history[-6:], ChatMessage("user", prepared.prompt)],
            self.settings.llm_max_output_tokens,
            self.settings.llm_temperature,
        )
        return prepared.sources, tokens, False

    def prepare(self, req: RagRequest) -> tuple[PreparedContext, int]:
        query = self._retrieval_query(req)
        flt = SearchFilter(principal=req.principal, kinds=req.item_kinds, project_ids=req.project_ids)
        affinity = self.recommender.affinity(req.tenant_id, req.principal) if req.personalize else None
        with Timer("rag_retrieve"):
            hits = self.retriever.retrieve(
                RetrievalRequest(req.tenant_id, query, flt, k=max(req.k * 2, 12), affinity=affinity)
            )
        s = self.settings
        relevant = [
            h
            for h in hits
            if h.signals.get("dense", 0.0) >= s.rag_min_dense_similarity
            or h.signals.get("coverage", 0.0) >= s.rag_min_lexical_coverage
        ]
        if not relevant:
            return PreparedContext("", [], True), len(hits)
        prompt, sources = self._assemble(req, relevant[: req.k * 2])
        return PreparedContext(prompt, sources, not sources), len(hits)

    def _retrieval_query(self, req: RagRequest) -> str:
        """Fold the latest user turns into the query so pronouns in follow ups resolve."""
        previous = [m.content for m in req.history if m.role == "user"][-2:]
        parts = [*previous, req.question]
        if req.context_item_id:
            stored = self.repo.get_item(req.tenant_id, req.context_item_id)
            if stored is not None:
                parts.append(stored.item.title)
        return "\n".join(p.strip() for p in parts if p and p.strip())

    def _assemble(self, req: RagRequest, hits: list[ScoredChunk]) -> tuple[str, list[Citation]]:
        items = self.repo.get_items(req.tenant_id, {h.chunk.item_id for h in hits})
        budget = self.settings.rag_context_token_budget
        used = 0
        blocks: list[str] = []
        sources: list[Citation] = []
        for hit in hits:
            stored = items.get(hit.chunk.item_id)
            if stored is None:
                continue
            item = stored.item
            text = strip_context_header(hit.chunk.text, item.title, hit.chunk.heading)
            cost = count_tokens(text) + 20
            if used + cost > budget and sources:
                continue
            index = len(sources) + 1
            meta = [item.kind.value]
            if item.project_id:
                meta.append(f"project {item.project_id}")
            meta.append(f"updated {item.updated_at.date().isoformat()}")
            heading = f" > {hit.chunk.heading}" if hit.chunk.heading else ""
            blocks.append(f"[{index}] {item.title}{heading} ({', '.join(meta)})\n{text}\n</source>")
            sources.append(
                Citation(
                    index=index,
                    item_id=item.id,
                    chunk_id=hit.chunk.chunk_id,
                    title=item.title,
                    url=item.url,
                    snippet=text[:300],
                    score=round(hit.score, 6),
                )
            )
            used += cost
            if len(sources) >= req.k:
                break
        prompt = USER_TEMPLATE.format(sources="\n\n".join(blocks), question=req.question.strip())
        return prompt, sources


def strip_context_header(text: str, title: str, heading: str | None) -> str:
    header = " > ".join(p for p in (title, heading) if p)
    if header and text.startswith(header + "\n"):
        return text[len(header) + 1 :]
    return text


def validate_citations(text: str, sources: list[Citation]) -> tuple[str, list[Citation]]:
    """Remove citations that point to non existent sources and return the sources used."""
    valid = {c.index: c for c in sources}
    used: dict[int, Citation] = {}

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index in valid:
            used[index] = valid[index]
            return match.group(0)
        return ""

    cleaned = _CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, [used[i] for i in sorted(used)]
