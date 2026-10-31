"""Persistence layer. The relational store is the system of record for Vantix.

Indexes (vector and lexical) are derived views that can always be rebuilt from this store,
which is what ``vantix reindex`` does.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import and_, create_engine, delete, event, func, insert, or_, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError

from vantix.domain.models import (
    AccessControl,
    Chunk,
    Interaction,
    Item,
    ItemKind,
    TaskStatus,
    ensure_utc,
    utcnow,
)
from vantix.storage import schema

_SQLITE_INIT_LOCK = threading.Lock()
_SCHEMA_LOCK_KEY = 7_302_118_446  # arbitrary constant identifying the Vantix schema lock


def encode_vector(vector: np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_vector(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


@dataclass(slots=True)
class StoredItem:
    item: Item
    content_hash: str
    embedding: np.ndarray | None
    deleted: bool


@dataclass(slots=True)
class JobRecord:
    job_id: str
    tenant_id: str
    status: str
    total: int
    processed: int
    failed: int
    errors: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class Repository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            path = database_url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if path == ":memory:" or database_url == "sqlite://":
                from sqlalchemy.pool import StaticPool

                kwargs["poolclass"] = StaticPool
        else:
            kwargs.update(pool_size=10, max_overflow=20)
        self.engine: Engine = create_engine(database_url, **kwargs)
        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", _sqlite_pragmas)

    # Schema ---------------------------------------------------------------------------
    def init_schema(self) -> None:
        """Create tables and indexes if missing.

        On PostgreSQL a transaction scoped advisory lock serialises concurrent callers, so many
        replicas (or a migration job racing a rollout) can start at once without
        "relation already exists" failures.
        """
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEMA_LOCK_KEY})
                schema.metadata.create_all(conn)
        else:
            # SQLite has no advisory locks: serialise creators within this process, and retry
            # transient contention from other processes ("already exists" after a lost race,
            # or "database is locked" while another connection holds the write lock).
            with _SQLITE_INIT_LOCK:
                for attempt in range(8):
                    try:
                        schema.metadata.create_all(self.engine)
                        return
                    except OperationalError as exc:
                        message = str(exc)
                        transient = "already exists" in message or "locked" in message
                        if not transient or attempt == 7:
                            raise
                        time.sleep(0.05 * (2**attempt))

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    # Items ----------------------------------------------------------------------------
    def get_item(self, tenant_id: str, item_id: str, *, include_deleted: bool = False) -> StoredItem | None:
        found = self.get_items(tenant_id, [item_id], include_deleted=include_deleted)
        return found.get(item_id)

    def get_items(
        self, tenant_id: str, item_ids: Iterable[str], *, include_deleted: bool = False
    ) -> dict[str, StoredItem]:
        ids = list(dict.fromkeys(item_ids))
        result: dict[str, StoredItem] = {}
        if not ids:
            return result
        t = schema.items
        with self.engine.connect() as conn:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                stmt = select(t).where(and_(t.c.tenant_id == tenant_id, t.c.item_id.in_(batch)))
                if not include_deleted:
                    stmt = stmt.where(t.c.deleted.is_(False))
                for row in conn.execute(stmt).mappings():
                    stored = _row_to_item(row)
                    result[stored.item.id] = stored
        return result

    def iter_items(
        self,
        tenant_id: str,
        *,
        kinds: list[ItemKind] | None = None,
        active_tasks_only: bool = False,
        batch_size: int = 1000,
    ) -> Iterator[StoredItem]:
        t = schema.items
        last_id = ""
        while True:
            stmt = (
                select(t)
                .where(and_(t.c.tenant_id == tenant_id, t.c.deleted.is_(False), t.c.item_id > last_id))
                .order_by(t.c.item_id)
                .limit(batch_size)
            )
            if kinds:
                stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
            if active_tasks_only:
                stmt = stmt.where(t.c.status.in_([s.value for s in TaskStatus if s.is_active]))
            with self.engine.connect() as conn:
                rows = list(conn.execute(stmt).mappings())
            if not rows:
                return
            for row in rows:
                yield _row_to_item(row)
            last_id = rows[-1]["item_id"]

    def upsert_item(
        self,
        tenant_id: str,
        item: Item,
        *,
        content_hash: str,
        embedding: np.ndarray | None,
        chunks: list[tuple[Chunk, np.ndarray | None]],
    ) -> None:
        """Insert or replace an item and all of its chunks in a single transaction.

        Two workers may race to insert the same new item. The loser hits a primary key
        violation, its transaction rolls back, and the retry takes the update path.
        """
        for attempt in range(3):
            try:
                self._upsert_item_once(tenant_id, item, content_hash, embedding, chunks)
                return
            except IntegrityError:
                if attempt == 2:
                    raise

    def _upsert_item_once(
        self,
        tenant_id: str,
        item: Item,
        content_hash: str,
        embedding: np.ndarray | None,
        chunks: list[tuple[Chunk, np.ndarray | None]],
    ) -> None:
        t = schema.items
        c = schema.chunks
        row = _item_to_row(tenant_id, item, content_hash, embedding)
        with self.engine.begin() as conn:
            exists = conn.execute(
                select(t.c.item_id).where(and_(t.c.tenant_id == tenant_id, t.c.item_id == item.id))
            ).first()
            if exists:
                conn.execute(
                    update(t).where(and_(t.c.tenant_id == tenant_id, t.c.item_id == item.id)).values(**row)
                )
            else:
                conn.execute(insert(t).values(**row))
            conn.execute(delete(c).where(and_(c.c.tenant_id == tenant_id, c.c.item_id == item.id)))
            if chunks:
                conn.execute(
                    insert(c),
                    [
                        {
                            "tenant_id": tenant_id,
                            "chunk_id": chunk.chunk_id,
                            "item_id": chunk.item_id,
                            "ordinal": chunk.ordinal,
                            "text": chunk.text,
                            "heading": chunk.heading,
                            "token_count": chunk.token_count,
                            "embedding": encode_vector(vec),
                        }
                        for chunk, vec in chunks
                    ],
                )

    def update_item_metadata_only(self, tenant_id: str, item: Item, content_hash: str) -> None:
        """Update mutable metadata (status, assignees, ACL) without touching chunks."""
        t = schema.items
        row = _item_to_row(tenant_id, item, content_hash, None)
        row.pop("embedding")
        with self.engine.begin() as conn:
            conn.execute(
                update(t).where(and_(t.c.tenant_id == tenant_id, t.c.item_id == item.id)).values(**row)
            )

    def chunk_ids_for_item(self, tenant_id: str, item_id: str) -> list[str]:
        c = schema.chunks
        with self.engine.connect() as conn:
            return [
                r[0]
                for r in conn.execute(
                    select(c.c.chunk_id).where(and_(c.c.tenant_id == tenant_id, c.c.item_id == item_id))
                )
            ]

    def mark_deleted(self, tenant_id: str, item_id: str) -> list[str]:
        t = schema.items
        c = schema.chunks
        chunk_ids = self.chunk_ids_for_item(tenant_id, item_id)
        with self.engine.begin() as conn:
            conn.execute(
                update(t)
                .where(and_(t.c.tenant_id == tenant_id, t.c.item_id == item_id))
                .values(deleted=True, updated_at=utcnow(), embedding=None)
            )
            conn.execute(delete(c).where(and_(c.c.tenant_id == tenant_id, c.c.item_id == item_id)))
        return chunk_ids

    def get_chunks(
        self, tenant_id: str, chunk_ids: Iterable[str]
    ) -> dict[str, tuple[str, int, str, str | None, int]]:
        """Return chunk_id to (item_id, ordinal, text, heading, token_count)."""
        ids = list(dict.fromkeys(chunk_ids))
        out: dict[str, tuple[str, int, str, str | None, int]] = {}
        c = schema.chunks
        with self.engine.connect() as conn:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                stmt = select(
                    c.c.chunk_id, c.c.item_id, c.c.ordinal, c.c.text, c.c.heading, c.c.token_count
                ).where(and_(c.c.tenant_id == tenant_id, c.c.chunk_id.in_(batch)))
                for r in conn.execute(stmt):
                    out[r[0]] = (r[1], r[2], r[3], r[4], r[5])
        return out

    def iter_chunk_rows(self, tenant_id: str, batch_size: int = 2000) -> Iterator[dict[str, Any]]:
        c = schema.chunks
        last = ""
        while True:
            stmt = (
                select(c)
                .where(and_(c.c.tenant_id == tenant_id, c.c.chunk_id > last))
                .order_by(c.c.chunk_id)
                .limit(batch_size)
            )
            with self.engine.connect() as conn:
                rows = [dict(r) for r in conn.execute(stmt).mappings()]
            if not rows:
                return
            yield from rows
            last = rows[-1]["chunk_id"]

    def list_tenants(self) -> list[str]:
        """Every tenant that has content or activity."""
        with self.engine.connect() as conn:
            found = {r[0] for r in conn.execute(select(schema.items.c.tenant_id).distinct())}
            found |= {r[0] for r in conn.execute(select(schema.interactions.c.tenant_id).distinct())}
        return sorted(found)

    def interaction_count(self, tenant_id: str) -> int:
        t = schema.interactions
        with self.engine.connect() as conn:
            return int(
                conn.execute(
                    select(func.count()).select_from(t).where(t.c.tenant_id == tenant_id)
                ).scalar_one()
            )

    # Interactions ---------------------------------------------------------------------
    def add_interactions(self, tenant_id: str, interactions: Iterable[Interaction]) -> int:
        """Insert interactions idempotently in bulk. Returns the number of new rows.

        Duplicates are detected both against stored rows and inside the batch itself, using one
        existence query per 500 events rather than one query per event.
        """
        t = schema.interactions
        unique = list({it.event_id: it for it in interactions}.values())
        inserted = 0
        with self.engine.begin() as conn:
            for start in range(0, len(unique), 500):
                batch = unique[start : start + 500]
                existing = {
                    r[0]
                    for r in conn.execute(
                        select(t.c.event_id).where(
                            and_(t.c.tenant_id == tenant_id, t.c.event_id.in_([i.event_id for i in batch]))
                        )
                    )
                }
                rows = [
                    {
                        "tenant_id": tenant_id,
                        "event_id": it.event_id,
                        "user_id": it.user_id,
                        "item_id": it.item_id,
                        "type": it.type.value,
                        "weight": it.weight,
                        "occurred_at": it.occurred_at,
                        "project_id": it.project_id,
                        "query": it.query,
                    }
                    for it in batch
                    if it.event_id not in existing
                ]
                if rows:
                    conn.execute(insert(t), rows)
                    inserted += len(rows)
        return inserted

    def delete_user_interactions(self, tenant_id: str, user_id: str) -> int:
        """Erase a user's interaction history (right to erasure). Returns rows removed."""
        t = schema.interactions
        with self.engine.begin() as conn:
            result = conn.execute(delete(t).where(and_(t.c.tenant_id == tenant_id, t.c.user_id == user_id)))
        return int(result.rowcount or 0)

    def user_interactions(
        self, tenant_id: str, user_id: str, *, limit: int = 200
    ) -> list[tuple[str, str, float, datetime]]:
        """Most recent interactions of a user as (item_id, type, weight, occurred_at)."""
        t = schema.interactions
        stmt = (
            select(t.c.item_id, t.c.type, t.c.weight, t.c.occurred_at)
            .where(and_(t.c.tenant_id == tenant_id, t.c.user_id == user_id))
            .order_by(t.c.occurred_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            return [(r[0], r[1], float(r[2]), ensure_utc(r[3])) for r in conn.execute(stmt)]  # type: ignore[misc]

    def iter_interactions(
        self, tenant_id: str, *, since: datetime | None = None, batch_size: int = 5000
    ) -> Iterator[tuple[str, str, str, float, datetime]]:
        """Yield (user_id, item_id, type, weight, occurred_at) in time order.

        Uses keyset pagination on (occurred_at, event_id), so each page is an index range scan
        and the total cost stays linear in the number of interactions.
        """
        t = schema.interactions
        last: tuple[datetime, str] | None = None
        while True:
            stmt = (
                select(t.c.user_id, t.c.item_id, t.c.type, t.c.weight, t.c.occurred_at, t.c.event_id)
                .where(t.c.tenant_id == tenant_id)
                .order_by(t.c.occurred_at, t.c.event_id)
                .limit(batch_size)
            )
            if since is not None:
                stmt = stmt.where(t.c.occurred_at >= since)
            if last is not None:
                stmt = stmt.where(
                    or_(
                        t.c.occurred_at > last[0],
                        and_(t.c.occurred_at == last[0], t.c.event_id > last[1]),
                    )
                )
            with self.engine.connect() as conn:
                rows = list(conn.execute(stmt))
            if not rows:
                return
            for r in rows:
                yield r[0], r[1], r[2], float(r[3]), ensure_utc(r[4])  # type: ignore[misc]
            last = (rows[-1][4], rows[-1][5])

    # Idempotency ----------------------------------------------------------------------
    def claim_event(self, tenant_id: str, event_id: str) -> bool:
        """Record an event as processed. Returns False if it was already processed."""
        t = schema.processed_events
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(t).values(tenant_id=tenant_id, event_id=event_id, processed_at=utcnow()))
            return True
        except IntegrityError:
            return False

    def release_events(self, tenant_id: str, event_ids: Iterable[str]) -> None:
        """Forget claimed events whose processing failed, so a retry is not treated as a duplicate."""
        ids = list(event_ids)
        if not ids:
            return
        t = schema.processed_events
        with self.engine.begin() as conn:
            for start in range(0, len(ids), 500):
                conn.execute(
                    delete(t).where(
                        and_(t.c.tenant_id == tenant_id, t.c.event_id.in_(ids[start : start + 500]))
                    )
                )

    # Jobs -----------------------------------------------------------------------------
    def create_job(self, tenant_id: str, total: int) -> str:
        job_id = uuid.uuid4().hex
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                insert(schema.jobs).values(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    status="queued",
                    total=total,
                    processed=0,
                    failed=0,
                    errors=[],
                    created_at=now,
                    updated_at=now,
                )
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        processed_delta: int = 0,
        failed_delta: int = 0,
        error: dict[str, Any] | None = None,
    ) -> None:
        t = schema.jobs
        with self.engine.begin() as conn:
            row = conn.execute(select(t).where(t.c.job_id == job_id)).mappings().first()
            if row is None:
                return
            errors = list(row["errors"] or [])
            if error is not None and len(errors) < 100:
                errors.append(error)
            values: dict[str, Any] = {
                "processed": row["processed"] + processed_delta,
                "failed": row["failed"] + failed_delta,
                "errors": errors,
                "updated_at": utcnow(),
            }
            if status is not None:
                values["status"] = status
            conn.execute(update(t).where(t.c.job_id == job_id).values(**values))

    def get_job(self, tenant_id: str, job_id: str) -> JobRecord | None:
        t = schema.jobs
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(t).where(and_(t.c.job_id == job_id, t.c.tenant_id == tenant_id)))
                .mappings()
                .first()
            )
        if row is None:
            return None
        return JobRecord(
            job_id=row["job_id"],
            tenant_id=row["tenant_id"],
            status=row["status"],
            total=row["total"],
            processed=row["processed"],
            failed=row["failed"],
            errors=list(row["errors"] or []),
            created_at=ensure_utc(row["created_at"]),  # type: ignore[arg-type]
            updated_at=ensure_utc(row["updated_at"]),  # type: ignore[arg-type]
        )

    # Tenant state ---------------------------------------------------------------------
    def _ensure_tenant_state(self, tenant_id: str) -> None:
        """Create the tenant state row if missing. Safe under concurrent callers."""
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(schema.tenant_state).values(tenant_id=tenant_id, content_version=0))
        except IntegrityError:
            pass  # another writer created it first

    def bump_content_version(self, tenant_id: str) -> int:
        """Atomically increment the tenant content version (used to invalidate caches)."""
        t = schema.tenant_state
        for _ in range(2):
            with self.engine.begin() as conn:
                result = conn.execute(
                    update(t)
                    .where(t.c.tenant_id == tenant_id)
                    .values(content_version=t.c.content_version + 1)
                )
                if result.rowcount:
                    version = conn.execute(
                        select(t.c.content_version).where(t.c.tenant_id == tenant_id)
                    ).scalar_one()
                    return int(version)
            self._ensure_tenant_state(tenant_id)
        raise RuntimeError(f"Could not update tenant state for {tenant_id}")

    def content_version(self, tenant_id: str) -> int:
        t = schema.tenant_state
        with self.engine.connect() as conn:
            row = conn.execute(select(t.c.content_version).where(t.c.tenant_id == tenant_id)).first()
        return int(row[0]) if row else 0

    def set_model_trained(self, tenant_id: str, when: datetime) -> None:
        t = schema.tenant_state
        self._ensure_tenant_state(tenant_id)
        with self.engine.begin() as conn:
            conn.execute(update(t).where(t.c.tenant_id == tenant_id).values(model_trained_at=when))

    def stats(self, tenant_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            items = conn.execute(
                select(schema.items.c.kind, func.count())
                .where(and_(schema.items.c.tenant_id == tenant_id, schema.items.c.deleted.is_(False)))
                .group_by(schema.items.c.kind)
            ).all()
            chunks = conn.execute(
                select(func.count()).select_from(schema.chunks).where(schema.chunks.c.tenant_id == tenant_id)
            ).scalar_one()
            inter = conn.execute(
                select(func.count())
                .select_from(schema.interactions)
                .where(schema.interactions.c.tenant_id == tenant_id)
            ).scalar_one()
            state = (
                conn.execute(select(schema.tenant_state).where(schema.tenant_state.c.tenant_id == tenant_id))
                .mappings()
                .first()
            )
        return {
            "items_by_kind": {k: int(n) for k, n in items},
            "chunks": int(chunks),
            "interactions": int(inter),
            "content_version": int(state["content_version"]) if state else 0,
            "model_trained_at": ensure_utc(state["model_trained_at"]) if state else None,
        }


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout=30000")
        # WAL is persistent in the database file; if another connection is switching it at
        # this moment the switch is already happening, so contention here is harmless.
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _item_to_row(
    tenant_id: str, item: Item, content_hash: str, embedding: np.ndarray | None
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "item_id": item.id,
        "kind": item.kind.value,
        "title": item.title,
        "body": item.body,
        "url": item.url,
        "project_id": item.project_id,
        "author_id": item.author_id,
        "parent_id": item.parent_id,
        "tags": item.tags,
        "status": item.status.value if item.status else None,
        "priority": item.priority,
        "assignee_ids": item.assignee_ids,
        "linked_item_ids": item.linked_item_ids,
        "due_at": item.due_at,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "language": item.language,
        "acl": item.acl.model_dump(mode="json"),
        "item_metadata": item.metadata,
        "content_hash": content_hash,
        "embedding": encode_vector(embedding),
        "deleted": False,
        "indexed_at": utcnow(),
    }


def _row_to_item(row: Any) -> StoredItem:
    item = Item(
        id=row["item_id"],
        kind=ItemKind(row["kind"]),
        title=row["title"] or "",
        body=row["body"] or "",
        url=row["url"],
        project_id=row["project_id"],
        author_id=row["author_id"],
        parent_id=row["parent_id"],
        tags=list(row["tags"] or []),
        status=TaskStatus(row["status"]) if row["status"] else None,
        priority=row["priority"],
        assignee_ids=list(row["assignee_ids"] or []),
        linked_item_ids=list(row["linked_item_ids"] or []),
        due_at=ensure_utc(row["due_at"]),
        created_at=ensure_utc(row["created_at"]),
        updated_at=ensure_utc(row["updated_at"]),
        language=row["language"] or "en",
        acl=AccessControl.model_validate(row["acl"] or {}),
        metadata=dict(row["item_metadata"] or {}),
    )
    return StoredItem(
        item=item,
        content_hash=row["content_hash"],
        embedding=decode_vector(row["embedding"]),
        deleted=bool(row["deleted"]),
    )
