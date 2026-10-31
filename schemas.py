"""Event contract between the host platform and Vantix.

Every change in the host platform is described by a small, versionless envelope. Delivery
is at least once: Vantix deduplicates on ``event_id``, so the host may retry freely.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from vantix.domain.models import InteractionType, Item, utcnow


class EventType(StrEnum):
    ITEM_UPSERTED = "item.upserted"
    ITEM_DELETED = "item.deleted"
    INTERACTION_RECORDED = "interaction.recorded"


class InteractionData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=256)
    item_id: str = Field(min_length=1, max_length=256)
    type: InteractionType
    project_id: str | None = None
    query: str | None = Field(default=None, max_length=2000)
    value: float | None = None


class DeletedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256)


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=256)
    occurred_at: datetime = Field(default_factory=utcnow)


class ItemUpsertedEvent(_Envelope):
    type: Literal["item.upserted"]
    data: Item


class ItemDeletedEvent(_Envelope):
    type: Literal["item.deleted"]
    data: DeletedData


class InteractionRecordedEvent(_Envelope):
    type: Literal["interaction.recorded"]
    data: InteractionData


Event = Annotated[
    ItemUpsertedEvent | ItemDeletedEvent | InteractionRecordedEvent, Field(discriminator="type")
]


class EventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[Event] = Field(min_length=1, max_length=1000)


class EventBatchResult(BaseModel):
    accepted: int
    duplicates: int
    interactions_recorded: int
    items_queued: int
    deletes_queued: int
    job_id: str | None = None
