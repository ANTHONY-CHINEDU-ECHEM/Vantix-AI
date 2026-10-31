"""Idempotent processing of host platform events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantix.domain.models import Interaction, Item
from vantix.events.schemas import (
    EventBatch,
    EventBatchResult,
    InteractionRecordedEvent,
    ItemDeletedEvent,
    ItemUpsertedEvent,
)
from vantix.observability.metrics import EVENTS_RECEIVED
from vantix.storage.repository import Repository

if TYPE_CHECKING:
    from vantix.ingestion.queue import IngestionQueue
    from vantix.recommender.scheduler import TrainingScheduler


class EventProcessor:
    def __init__(self, repository: Repository, queue: IngestionQueue, scheduler: TrainingScheduler) -> None:
        self.repo = repository
        self.queue = queue
        self.scheduler = scheduler

    def process(self, tenant_id: str, batch: EventBatch) -> EventBatchResult:
        upserts: dict[str, Item] = {}
        deletes: dict[str, None] = {}
        interactions: list[Interaction] = []
        duplicates = 0
        claimed: list[str] = []

        for event in sorted(batch.events, key=lambda e: e.occurred_at):
            if isinstance(event, InteractionRecordedEvent):
                d = event.data
                interactions.append(
                    Interaction(
                        event_id=event.event_id,
                        user_id=d.user_id,
                        item_id=d.item_id,
                        type=d.type,
                        occurred_at=event.occurred_at,
                        project_id=d.project_id,
                        query=d.query,
                        value=d.value,
                    )
                )
                continue
            if not self.repo.claim_event(tenant_id, event.event_id):
                duplicates += 1
                EVENTS_RECEIVED.labels(type=event.type, outcome="duplicate").inc()
                continue
            claimed.append(event.event_id)
            EVENTS_RECEIVED.labels(type=event.type, outcome="accepted").inc()
            if isinstance(event, ItemUpsertedEvent):
                deletes.pop(event.data.id, None)
                upserts[event.data.id] = event.data
            elif isinstance(event, ItemDeletedEvent):
                upserts.pop(event.data.id, None)
                deletes[event.data.id] = None

        recorded = self.repo.add_interactions(tenant_id, interactions) if interactions else 0
        interaction_duplicates = len(interactions) - recorded
        EVENTS_RECEIVED.labels(type="interaction.recorded", outcome="accepted").inc(recorded)
        if interaction_duplicates:
            EVENTS_RECEIVED.labels(type="interaction.recorded", outcome="duplicate").inc(
                interaction_duplicates
            )
        if recorded:
            self.scheduler.mark_dirty(tenant_id)

        job_id = None
        if upserts or deletes:
            try:
                job_id = self.queue.submit(tenant_id, list(upserts.values()), list(deletes))
            except Exception:
                # The work never reached the queue: release the claims so that the host's retry
                # of this batch is processed instead of being discarded as a duplicate.
                self.repo.release_events(tenant_id, claimed)
                raise

        total_duplicates = duplicates + interaction_duplicates
        return EventBatchResult(
            accepted=len(batch.events) - total_duplicates,
            duplicates=total_duplicates,
            interactions_recorded=recorded,
            items_queued=len(upserts),
            deletes_queued=len(deletes),
            job_id=job_id,
        )
