"""FastAPI application factory and HTTP routes (API version 1)."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from vantix import __version__
from vantix.api.schemas import (
    AskRequest,
    AskResponse,
    AssistRequestBody,
    AssistResponseBody,
    ErrorBody,
    FeedbackRequest,
    FeedbackResponse,
    ItemResult,
    JobResponse,
    RecommendRequest,
    RecommendResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UpsertItemsRequest,
    UpsertItemsResponse,
)
from vantix.assist import AssistRequest
from vantix.config import Settings, get_settings
from vantix.container import Container
from vantix.domain.models import Interaction, InteractionType, Item
from vantix.errors import NotFoundError, ValidationFailed, VantixError
from vantix.events.schemas import EventBatch, EventBatchResult
from vantix.index.base import SearchFilter
from vantix.observability.logging import (
    configure_logging,
    get_logger,
    request_id_var,
    tenant_id_var,
)
from vantix.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS, REGISTRY
from vantix.rag.llm import ChatMessage
from vantix.rag.service import RagRequest, strip_context_header, validate_citations
from vantix.recommender.engine import RecommendationContext, RecommendationQuery
from vantix.retrieval.hybrid import RetrievalRequest
from vantix.security.auth import ApiKeyAuthenticator, CallerIdentity, RateLimiter, verify_signature

log = get_logger(__name__)

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_.:]{1,64}")

_FEEDBACK_TYPES = {
    ("click", "search"): InteractionType.SEARCH_CLICK,
    ("click", "recommendation"): InteractionType.RECOMMENDATION_CLICK,
    ("click", "answer"): InteractionType.SEARCH_CLICK,
    ("click", "assist"): InteractionType.RECOMMENDATION_CLICK,
    ("upvote", None): InteractionType.UPVOTE,
    ("downvote", None): InteractionType.DOWNVOTE,
    ("dismiss", None): InteractionType.DISMISS,
    ("bookmark", None): InteractionType.BOOKMARK,
}


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = container or Container.build(settings)
        built.start()
        app.state.container = built
        log.info("vantix_started", extra={"version": __version__, "environment": settings.environment})
        try:
            yield
        finally:
            built.stop()
            log.info("vantix_stopped")

    app = FastAPI(
        title="Vantix AI",
        version=__version__,
        description=(
            "Hybrid recommender and retrieval augmented generation engine for project management platforms."
        ),
        lifespan=lifespan,
        responses={401: {"model": ErrorBody}, 403: {"model": ErrorBody}, 429: {"model": ErrorBody}},
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )

    authenticator = ApiKeyAuthenticator(settings.tenant_keys(), settings.admin_keys())
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get("x-request-id", "")
        # Accept a caller supplied correlation id only if it is short and safe to log.
        rid = supplied if _REQUEST_ID_RE.fullmatch(supplied) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
            request_id_var.reset(token)

    @app.exception_handler(VantixError)
    async def handle_error(_request: Request, exc: VantixError) -> JSONResponse:
        body = ErrorBody(
            code=exc.code, message=exc.message, details=exc.details, request_id=request_id_var.get()
        )
        headers = {}
        if "retry_after_seconds" in exc.details:
            headers["Retry-After"] = str(int(float(exc.details["retry_after_seconds"])) + 1)  # type: ignore[arg-type]
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)

    def c(request: Request) -> Container:
        return request.app.state.container  # type: ignore[no-any-return]

    def identity(authorization: str | None = Header(default=None)) -> CallerIdentity:
        ident = authenticator.authenticate(authorization)
        limiter.check(ident.key_fingerprint)
        tenant_id_var.set(ident.tenant_id)
        return ident

    def admin_tenant(ident: CallerIdentity, tenant_id: str | None) -> str:
        if tenant_id and tenant_id != ident.tenant_id:
            ApiKeyAuthenticator.require_admin(ident)
            return tenant_id
        return ident.tenant_id

    # Health -------------------------------------------------------------------------
    @app.get("/v1/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/ready", tags=["operations"])
    def ready(request: Request) -> JSONResponse:
        ctr = c(request)
        checks = {"vector_index": ctr.vector_index.healthy(), "lexical_index": ctr.lexical_index.healthy()}
        try:
            checks["database"] = ctr.repository.ping()
        except Exception:  # noqa: BLE001
            checks["database"] = False
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})

    @app.get("/v1/metrics", tags=["operations"], response_class=PlainTextResponse)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # Items ----------------------------------------------------------------------------
    @app.post("/v1/items", response_model=UpsertItemsResponse, tags=["ingestion"], status_code=202)
    def upsert_items(body: UpsertItemsRequest, request: Request, ident: CallerIdentity = Depends(identity)):  # type: ignore[no-untyped-def]
        ctr = c(request)
        if body.wait:
            results = ctr.ingestion.ingest(ident.tenant_id, body.items)
            ctr.scheduler.mark_dirty(ident.tenant_id)
            return JSONResponse(
                status_code=200,
                content=UpsertItemsResponse(
                    results=[
                        ItemResult(item_id=r.item_id, status=r.status, chunks=r.chunks, error=r.error)
                        for r in results
                    ]
                ).model_dump(),
            )
        job_id = ctr.queue.submit(ident.tenant_id, body.items)
        return UpsertItemsResponse(job_id=job_id)

    @app.get("/v1/items/{item_id}", response_model=Item, tags=["ingestion"])
    def get_item(item_id: str, request: Request, ident: CallerIdentity = Depends(identity)) -> Item:
        stored = c(request).repository.get_item(ident.tenant_id, item_id)
        if stored is None:
            raise NotFoundError(f"Item {item_id} not found")
        return stored.item

    @app.delete("/v1/items/{item_id}", tags=["ingestion"])
    def delete_item(
        item_id: str, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> dict[str, object]:
        result = c(request).ingestion.delete(ident.tenant_id, item_id)
        return {"item_id": item_id, "status": result.status, "chunks_removed": result.chunks}

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse, tags=["ingestion"])
    def get_job(job_id: str, request: Request, ident: CallerIdentity = Depends(identity)) -> JobResponse:
        job = c(request).repository.get_job(ident.tenant_id, job_id)
        if job is None:
            raise NotFoundError(f"Job {job_id} not found")
        return JobResponse(**{name: getattr(job, name) for name in JobResponse.model_fields})

    # Events ---------------------------------------------------------------------------
    @app.post("/v1/events", response_model=EventBatchResult, tags=["events"], status_code=202)
    async def ingest_events(
        request: Request,
        ident: CallerIdentity = Depends(identity),
        signature: str | None = Header(default=None),
    ) -> EventBatchResult:
        raw = await request.body()
        if settings.webhook_secret:
            verify_signature(settings.webhook_secret, raw, signature, settings.webhook_tolerance_seconds)
        try:
            batch = EventBatch.model_validate_json(raw)
        except ValidationError as exc:
            raise ValidationFailed(
                "Invalid event batch",
                details={"errors": json.loads(exc.json(include_url=False))[:20]},
            ) from exc
        return c(request).events.process(ident.tenant_id, batch)

    # Retrieval ------------------------------------------------------------------------
    @app.post("/v1/search", response_model=SearchResponse, tags=["retrieval"])
    def search(
        body: SearchRequest, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> SearchResponse:
        ctr = c(request)
        flt = SearchFilter(
            principal=body.principal,
            kinds=body.item_kinds,
            project_ids=body.project_ids,
            active_tasks_only=body.active_tasks_only,
            updated_after=body.updated_after,
        )
        affinity = ctr.recommender.affinity(ident.tenant_id, body.principal) if body.personalize else None
        hits = ctr.retriever.retrieve(
            RetrievalRequest(ident.tenant_id, body.query, flt, k=body.k, affinity=affinity)
        )
        items = ctr.repository.get_items(ident.tenant_id, {h.chunk.item_id for h in hits})
        out: list[SearchHit] = []
        for h in hits:
            stored = items.get(h.chunk.item_id)
            if stored is None:
                continue
            out.append(
                SearchHit(
                    item_id=h.chunk.item_id,
                    chunk_id=h.chunk.chunk_id,
                    kind=stored.item.kind,
                    title=stored.item.title,
                    url=stored.item.url,
                    project_id=stored.item.project_id,
                    snippet=strip_context_header(h.chunk.text, stored.item.title, h.chunk.heading)[:400],
                    heading=h.chunk.heading,
                    score=round(h.score, 6),
                    signals=h.signals,
                )
            )
        return SearchResponse(hits=out)

    @app.post("/v1/ask", response_model=AskResponse, tags=["retrieval"])
    def ask(body: AskRequest, request: Request, ident: CallerIdentity = Depends(identity)):  # type: ignore[no-untyped-def]
        ctr = c(request)
        rag_request = RagRequest(
            tenant_id=ident.tenant_id,
            principal=body.principal,
            question=body.question,
            history=[ChatMessage(t.role, t.content) for t in body.history],
            project_ids=body.project_ids,
            item_kinds=body.item_kinds,
            context_item_id=body.context_item_id,
            k=body.k,
            personalize=body.personalize,
        )
        if body.stream:
            sources, tokens, abstained = ctr.rag.stream(rag_request)
            return StreamingResponse(
                _sse(sources, tokens, abstained, ctr.llm.name), media_type="text/event-stream"
            )
        result = ctr.rag.answer(rag_request)
        return AskResponse(
            answer=result.answer,
            citations=result.citations,
            grounded=result.grounded,
            abstained=result.abstained,
            model=result.model,
            usage=result.usage,
        )

    # Recommendations ------------------------------------------------------------------
    @app.post("/v1/recommendations", response_model=RecommendResponse, tags=["recommendations"])
    def recommend(
        body: RecommendRequest, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> RecommendResponse:
        ctr = c(request)
        recs = ctr.recommender.recommend(
            RecommendationQuery(
                tenant_id=ident.tenant_id,
                principal=body.principal,
                kind=body.kind,
                k=body.k,
                context=RecommendationContext(
                    body.context_item_id, body.context_query, body.context_project_id
                ),
                item_kinds=body.item_kinds,
                project_ids=body.project_ids,
                exclude_ids=set(body.exclude_ids),
                candidate_user_ids=body.candidate_user_ids,
                include_assigned_to_others=body.include_assigned_to_others,
                include_self=body.include_self,
                diversify=body.diversify,
            )
        )
        model = ctr.registry.get(ident.tenant_id)
        trained = model.trained_at if model.trained_at.year > 1970 else None
        return RecommendResponse(kind=body.kind, recommendations=recs, model_trained_at=trained)

    @app.post("/v1/assist", response_model=AssistResponseBody, tags=["recommendations"])
    def assist(
        body: AssistRequestBody, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> AssistResponseBody:
        result = c(request).assist.assist(
            AssistRequest(
                ident.tenant_id, body.principal, body.item_id, body.question, body.project_id, body.k
            )
        )
        answer = None
        if result.answer is not None:
            a = result.answer
            answer = AskResponse(
                answer=a.answer,
                citations=a.citations,
                grounded=a.grounded,
                abstained=a.abstained,
                model=a.model,
                usage=a.usage,
            )
        return AssistResponseBody(
            answer=answer,
            documents=result.documents,
            similar_tasks=result.similar_tasks,
            experts=result.experts,
        )

    @app.post("/v1/feedback", response_model=FeedbackResponse, tags=["recommendations"])
    def feedback(
        body: FeedbackRequest, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> FeedbackResponse:
        ctr = c(request)
        itype = _FEEDBACK_TYPES.get((body.signal, body.surface)) or _FEEDBACK_TYPES[(body.signal, None)]
        inserted = ctr.repository.add_interactions(
            ident.tenant_id,
            [
                Interaction(
                    event_id=body.event_id or f"fb_{uuid.uuid4().hex}",
                    user_id=body.principal.user_id,
                    item_id=body.target_id,
                    type=itype,
                    query=body.query,
                )
            ],
        )
        if inserted:
            ctr.scheduler.mark_dirty(ident.tenant_id)
        return FeedbackResponse(recorded=bool(inserted))

    @app.delete("/v1/users/{user_id}/interactions", tags=["privacy"])
    def erase_user(
        user_id: str, request: Request, ident: CallerIdentity = Depends(identity)
    ) -> dict[str, object]:
        """Right to erasure: delete every interaction of a user in the caller's tenant.

        The user disappears from collaborative and graph signals at the next training run, which
        this call schedules immediately.
        """
        ctr = c(request)
        removed = ctr.repository.delete_user_interactions(ident.tenant_id, user_id)
        ctr.scheduler.mark_dirty(ident.tenant_id)
        return {"user_id": user_id, "interactions_deleted": removed}

    # Administration -------------------------------------------------------------------
    @app.post("/v1/admin/train", tags=["administration"])
    def train(
        request: Request,
        ident: CallerIdentity = Depends(identity),
        tenant_id: str | None = Query(default=None),
    ) -> dict[str, object]:
        tenant = admin_tenant(ident, tenant_id)
        ctr = c(request)
        ctr.scheduler.train_now(tenant)
        model = ctr.registry.get(tenant)
        return {
            "tenant_id": tenant,
            "trained_at": model.trained_at.isoformat(),
            "users": len(model.user_ids),
            "items": len(model.item_ids),
            "graph_nodes": model.graph.size if model.graph else 0,
            "collaborative_model": model.als is not None,
        }

    @app.post("/v1/admin/reindex", tags=["administration"])
    def reindex(
        request: Request,
        ident: CallerIdentity = Depends(identity),
        tenant_id: str | None = Query(default=None),
        full: bool = Query(default=False, description="Rechunk and re embed every item"),
    ) -> dict[str, object]:
        tenant = admin_tenant(ident, tenant_id)
        ctr = c(request)
        if full:
            return {"tenant_id": tenant, "items_reembedded": ctr.ingestion.reembed_all(tenant)}
        count = ctr.ingestion.rebuild_indexes(tenant)
        ctr.repository.bump_content_version(tenant)
        return {"tenant_id": tenant, "chunks_indexed": count}

    @app.get("/v1/admin/stats", tags=["administration"])
    def stats(
        request: Request,
        ident: CallerIdentity = Depends(identity),
        tenant_id: str | None = Query(default=None),
    ) -> dict[str, object]:
        tenant = admin_tenant(ident, tenant_id)
        ctr = c(request)
        data = ctr.repository.stats(tenant)
        data["vector_index_chunks"] = ctr.vector_index.count(tenant)
        data["lexical_index_chunks"] = ctr.lexical_index.count(tenant)
        data["embedding_model"] = ctr.embedder.name
        data["llm"] = ctr.llm.name
        data["reranker"] = ctr.reranker.name
        return {"tenant_id": tenant, **data}

    return app


def _sse(sources, tokens: Iterator[str], abstained: bool, model: str) -> Iterator[bytes]:  # type: ignore[no-untyped-def]
    def event(name: str, data: object) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n".encode()

    yield event("sources", [s.model_dump() for s in sources])
    parts: list[str] = []
    try:
        for token in tokens:
            parts.append(token)
            yield event("token", {"text": token})
    except Exception as exc:  # noqa: BLE001
        log.exception("stream_failed")
        yield event("error", {"message": str(exc)})
        return
    text, used = validate_citations("".join(parts), sources)
    yield event(
        "done",
        {
            "answer": text,
            "citations": [u.model_dump() for u in used],
            "grounded": bool(used),
            "abstained": abstained,
            "model": model,
        },
    )


def app_factory() -> FastAPI:
    """Entry point for ASGI servers: ``uvicorn vantix.api.app:app_factory --factory``."""
    return create_app()
