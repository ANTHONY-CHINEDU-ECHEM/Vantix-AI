"""Hybrid retrieval: dense and lexical recall, weighted RRF fusion, personalisation,
second stage reranking, per item capping and MMR diversification.

Stages and their purpose:

1. Dense recall finds paraphrases and conceptual matches.
2. Lexical recall finds exact identifiers, acronyms, error codes and names.
3. Weighted reciprocal rank fusion merges both without score calibration.
4. Personal affinity (from the recommender) nudges items the user already works with.
5. A reranker rescores the head of the list with a stronger model.
6. Per item capping and MMR stop one long document from crowding out everything else.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vantix.cache import TTLCache
from vantix.config import Settings
from vantix.domain.models import ScoredChunk
from vantix.embeddings.base import Embedder
from vantix.index.base import IndexHit, LexicalIndex, SearchFilter, VectorIndex
from vantix.observability.metrics import Timer
from vantix.retrieval.fusion import jaccard, min_max, mmr, weighted_rrf
from vantix.retrieval.rerankers import NoopReranker, Reranker
from vantix.storage.repository import Repository
from vantix.text import tokenize


@dataclass(slots=True)
class RetrievalRequest:
    tenant_id: str
    query: str
    flt: SearchFilter
    k: int = 10
    affinity: dict[str, float] | None = None  # item_id to [0, 1]
    diversify: bool = True
    rerank: bool = True


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        reranker: Reranker,
        repository: Repository,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vectors = vector_index
        self.lexical = lexical_index
        self.reranker = reranker
        self.repo = repository
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vantix_recall")
        self._cache: TTLCache[list[ScoredChunk]] = TTLCache("retrieval", settings.retrieval_cache_ttl_seconds)

    def retrieve(self, request: RetrievalRequest) -> list[ScoredChunk]:
        key = self._cache_key(request)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._retrieve(request)
        self._cache.set(key, result)
        return result

    def _retrieve(self, req: RetrievalRequest) -> list[ScoredChunk]:
        s = self.settings
        query = req.query.strip()
        if not query:
            return []

        with Timer("recall"):
            dense_future = self._pool.submit(self._dense, req.tenant_id, query, s.retrieval_dense_k, req.flt)
            sparse_future = self._pool.submit(
                self.lexical.search, req.tenant_id, query, s.retrieval_sparse_k, req.flt
            )
            dense_hits: list[IndexHit] = dense_future.result()
            sparse_hits: list[IndexHit] = sparse_future.result()

        chunks = {h.chunk.chunk_id: h.chunk for h in dense_hits}
        chunks.update({h.chunk.chunk_id: h.chunk for h in sparse_hits})
        if not chunks:
            return []

        dense_scores = {h.chunk.chunk_id: h.score for h in dense_hits}
        sparse_scores = {h.chunk.chunk_id: h.score for h in sparse_hits}
        rankings = {
            "dense": [h.chunk.chunk_id for h in dense_hits],
            "sparse": [h.chunk.chunk_id for h in sparse_hits],
        }
        weights = {"dense": s.retrieval_dense_weight, "sparse": s.retrieval_sparse_weight}
        fused = weighted_rrf(rankings, weights, k=s.retrieval_fusion_k)
        max_rrf = sum(weights.values()) / (s.retrieval_fusion_k + 1)
        fused = {cid: score / max_rrf for cid, score in fused.items()}

        affinity = req.affinity or {}
        if affinity and s.retrieval_personal_weight > 0:
            for cid in fused:
                fused[cid] += s.retrieval_personal_weight * affinity.get(chunks[cid].item_id, 0.0)

        q_terms = set(tokenize(query))
        sparse_norm = min_max(sparse_scores)
        ordered = sorted(fused, key=fused.__getitem__, reverse=True)

        rerank_scores: dict[str, float] = {}
        if req.rerank and not isinstance(self.reranker, NoopReranker):
            head = ordered[: s.reranker_top_n]
            with Timer("rerank"):
                scores = self.reranker.score(query, [chunks[cid].text for cid in head])
            rerank_scores = dict(zip(head, scores, strict=True))

        results: list[ScoredChunk] = []
        for cid in ordered:
            chunk = chunks[cid]
            fused_score = fused[cid]
            rr = rerank_scores.get(cid)
            final = fused_score if rr is None else 0.4 * min(1.0, fused_score) + 0.6 * rr
            terms = set(tokenize(chunk.text))
            coverage = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            relevance = max(max(0.0, dense_scores.get(cid, 0.0)), coverage)
            signals = {
                "fused": round(fused_score, 6),
                "dense": round(dense_scores.get(cid, 0.0), 6),
                "sparse": round(sparse_norm.get(cid, 0.0), 6),
                "coverage": round(coverage, 6),
                "relevance": round(relevance, 6),
            }
            if rr is not None:
                signals["rerank"] = round(rr, 6)
            if affinity:
                signals["affinity"] = round(affinity.get(chunk.item_id, 0.0), 6)
            results.append(ScoredChunk(chunk=chunk, score=final, signals=signals))
        results.sort(key=lambda r: r.score, reverse=True)

        capped = self._cap_per_item(results, s.retrieval_max_chunks_per_document)
        if req.diversify and len(capped) > req.k:
            token_sets = {r.chunk.chunk_id: set(tokenize(r.chunk.text)) for r in capped[: req.k * 4]}
            capped = mmr(
                capped[: req.k * 4],
                relevance=lambda r: r.score,
                similarity=lambda a, b: jaccard(token_sets[a.chunk.chunk_id], token_sets[b.chunk.chunk_id]),
                k=req.k,
                lambda_=s.retrieval_mmr_lambda,
            )
        return capped[: req.k]

    def _dense(self, tenant_id: str, query: str, k: int, flt: SearchFilter) -> list[IndexHit]:
        vector = self.embedder.embed_query(query)
        return self.vectors.search(tenant_id, vector, k, flt)

    @staticmethod
    def _cap_per_item(results: list[ScoredChunk], cap: int) -> list[ScoredChunk]:
        counts: dict[str, int] = {}
        out: list[ScoredChunk] = []
        for r in results:
            n = counts.get(r.chunk.item_id, 0)
            if n < cap:
                out.append(r)
                counts[r.chunk.item_id] = n + 1
        return out

    def _cache_key(self, req: RetrievalRequest) -> str:
        flt = req.flt
        payload = {
            "t": req.tenant_id,
            "v": self.repo.content_version(req.tenant_id),
            "q": req.query,
            "k": req.k,
            "p": flt.principal.model_dump() if flt.principal else None,
            "kinds": [k.value for k in flt.kinds] if flt.kinds else None,
            "proj": flt.project_ids,
            "ids": flt.item_ids,
            "ex": sorted(flt.exclude_item_ids),
            "act": flt.active_tasks_only,
            "upd": flt.updated_after.isoformat() if flt.updated_after else None,
            "aff": sorted((req.affinity or {}).items()),
            "div": req.diversify,
            "rr": req.rerank,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
