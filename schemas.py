"""Public request and response models for the HTTP API (version 1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vantix.domain.models import Citation, Item, ItemKind, Principal, Recommendation
from vantix.recommender.engine import RecommendationKind


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpsertItemsRequest(_Strict):
    items: list[Item] = Field(min_length=1, max_length=1000)
    wait: bool = Field(default=False, description="Index synchronously instead of queuing a background job")


class ItemResult(BaseModel):
    item_id: str
    status: str
    chunks: int = 0
    error: str | None = None


class UpsertItemsResponse(BaseModel):
    job_id: str | None = None
    results: list[ItemResult] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    failed: int
    errors: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class SearchRequest(_Strict):
    principal: Principal
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)
    item_kinds: list[ItemKind] | None = None
    project_ids: list[str] | None = None
    active_tasks_only: bool = False
    updated_after: datetime | None = None
    personalize: bool = True


class SearchHit(BaseModel):
    item_id: str
    chunk_id: str
    kind: ItemKind
    title: str
    url: str | None
    project_id: str | None
    snippet: str
    heading: str | None
    score: float
    signals: dict[str, float]


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class HistoryTurn(_Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(_Strict):
    principal: Principal
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    project_ids: list[str] | None = None
    item_kinds: list[ItemKind] | None = None
    context_item_id: str | None = None
    k: int = Field(default=8, ge=1, le=20)
    personalize: bool = True
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    abstained: bool
    model: str
    usage: dict[str, int]


class RecommendRequest(_Strict):
    principal: Principal
    kind: RecommendationKind
    k: int = Field(default=10, ge=1, le=100)
    context_item_id: str | None = None
    context_query: str | None = Field(default=None, max_length=2000)
    context_project_id: str | None = None
    item_kinds: list[ItemKind] | None = None
    project_ids: list[str] | None = None
    exclude_ids: list[str] = Field(default_factory=list, max_length=1000)
    candidate_user_ids: list[str] | None = None
    include_assigned_to_others: bool = False
    include_self: bool = False
    diversify: bool = True


class RecommendResponse(BaseModel):
    kind: RecommendationKind
    recommendations: list[Recommendation]
    model_trained_at: datetime | None = None


class AssistRequestBody(_Strict):
    principal: Principal
    item_id: str | None = None
    question: str | None = Field(default=None, max_length=4000)
    project_id: str | None = None
    k: int = Field(default=5, ge=1, le=20)


class AssistResponseBody(BaseModel):
    answer: AskResponse | None
    documents: list[Recommendation]
    similar_tasks: list[Recommendation]
    experts: list[Recommendation]


class FeedbackRequest(_Strict):
    principal: Principal
    target_id: str
    signal: Literal["click", "upvote", "downvote", "dismiss", "bookmark"]
    surface: Literal["recommendation", "search", "answer", "assist"] = "recommendation"
    query: str | None = None
    event_id: str | None = None


class FeedbackResponse(BaseModel):
    recorded: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
