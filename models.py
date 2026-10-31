"""Domain model shared by every Vantix subsystem.

The model is deliberately platform neutral. A project management platform maps its own
entities (tasks, wiki pages, comments, files, messages, decisions) onto ``Item`` and its
user activity onto ``Interaction``. Everything else in Vantix is derived from those two.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ItemKind(StrEnum):
    TASK = "task"
    DOCUMENT = "document"
    WIKI_PAGE = "wiki_page"
    COMMENT = "comment"
    FILE = "file"
    MESSAGE = "message"
    DECISION = "decision"

    @property
    def is_knowledge(self) -> bool:
        return self is not ItemKind.TASK


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        return self not in (TaskStatus.DONE, TaskStatus.CANCELLED)


class Visibility(StrEnum):
    TENANT = "tenant"
    PROJECT = "project"
    RESTRICTED = "restricted"


class AccessControl(BaseModel):
    """Who may see an item. Enforced at retrieval time and again before any response."""

    model_config = ConfigDict(extra="forbid")

    visibility: Visibility = Visibility.PROJECT
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_group_ids: list[str] = Field(default_factory=list)


class Principal(BaseModel):
    """The end user on whose behalf a request is made. Supplied by the host platform."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=256)
    group_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    is_admin: bool = False


class Item(BaseModel):
    """A unit of knowledge or work: a task, a document, a wiki page, a comment and so on."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    kind: ItemKind = ItemKind.DOCUMENT
    title: str = Field(default="", max_length=1000)
    body: str = Field(default="", max_length=5_000_000)
    url: str | None = None
    project_id: str | None = None
    author_id: str | None = None
    parent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: TaskStatus | None = None
    priority: int | None = Field(default=None, ge=0, le=4)
    assignee_ids: list[str] = Field(default_factory=list)
    linked_item_ids: list[str] = Field(default_factory=list)
    due_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    language: str = "en"
    acl: AccessControl = Field(default_factory=AccessControl)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _normalise_tags(cls, tags: list[str]) -> list[str]:
        seen: list[str] = []
        for tag in tags:
            clean = tag.strip().lower()
            if clean and clean not in seen:
                seen.append(clean)
        return seen

    @field_validator("created_at", "updated_at", "due_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


class InteractionType(StrEnum):
    VIEW = "view"
    OPEN = "open"
    SEARCH_CLICK = "search_click"
    RECOMMENDATION_CLICK = "recommendation_click"
    CREATE = "create"
    EDIT = "edit"
    COMMENT = "comment"
    ASSIGN = "assign"
    COMPLETE = "complete"
    BOOKMARK = "bookmark"
    SHARE = "share"
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    DISMISS = "dismiss"


# Implicit feedback strength. Negative weights are used to suppress items, never to train ALS.
INTERACTION_WEIGHTS: dict[InteractionType, float] = {
    InteractionType.VIEW: 1.0,
    InteractionType.OPEN: 1.0,
    InteractionType.SEARCH_CLICK: 2.0,
    InteractionType.RECOMMENDATION_CLICK: 2.0,
    InteractionType.CREATE: 5.0,
    InteractionType.EDIT: 4.0,
    InteractionType.COMMENT: 3.0,
    InteractionType.ASSIGN: 4.0,
    InteractionType.COMPLETE: 5.0,
    InteractionType.BOOKMARK: 4.0,
    InteractionType.SHARE: 3.0,
    InteractionType.UPVOTE: 3.0,
    InteractionType.DOWNVOTE: -3.0,
    InteractionType.DISMISS: -2.0,
}


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=256)
    item_id: str = Field(min_length=1, max_length=256)
    type: InteractionType
    occurred_at: datetime = Field(default_factory=utcnow)
    project_id: str | None = None
    query: str | None = None
    value: float | None = None

    @field_validator("occurred_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)  # type: ignore[return-value]

    @property
    def weight(self) -> float:
        return self.value if self.value is not None else INTERACTION_WEIGHTS[self.type]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    chunk_id: str
    item_id: str
    tenant_id: str
    ordinal: int
    text: str
    heading: str | None = None
    token_count: int = 0
    # Denormalised payload used for filtering inside external indexes.
    kind: ItemKind = ItemKind.DOCUMENT
    project_id: str | None = None
    visibility: Visibility = Visibility.PROJECT
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_group_ids: list[str] = Field(default_factory=list)
    author_id: str | None = None
    status: TaskStatus | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    signals: dict[str, float] = Field(default_factory=dict)


class Reason(BaseModel):
    code: str
    message: str
    weight: float = 0.0


class Recommendation(BaseModel):
    target_id: str
    target_type: str
    score: float
    title: str | None = None
    url: str | None = None
    reasons: list[Reason] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    item_id: str
    chunk_id: str
    title: str
    url: str | None = None
    snippet: str
    score: float
