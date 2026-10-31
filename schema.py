"""Relational schema. Portable across SQLite (development) and PostgreSQL (production)."""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

items = Table(
    "vx_items",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("item_id", String(256), primary_key=True),
    Column("kind", String(32), nullable=False),
    Column("title", Text, nullable=False, default=""),
    Column("body", Text, nullable=False, default=""),
    Column("url", Text),
    Column("project_id", String(256)),
    Column("author_id", String(256)),
    Column("parent_id", String(256)),
    Column("tags", JSON, nullable=False, default=list),
    Column("status", String(32)),
    Column("priority", Integer),
    Column("assignee_ids", JSON, nullable=False, default=list),
    Column("linked_item_ids", JSON, nullable=False, default=list),
    Column("due_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("language", String(16), nullable=False, default="en"),
    Column("acl", JSON, nullable=False),
    Column("item_metadata", JSON, nullable=False, default=dict),
    Column("content_hash", String(64), nullable=False),
    Column("embedding", LargeBinary),
    Column("deleted", Boolean, nullable=False, default=False),
    Column("indexed_at", DateTime(timezone=True)),
)
Index("ix_vx_items_project", items.c.tenant_id, items.c.project_id)
Index("ix_vx_items_kind_status", items.c.tenant_id, items.c.kind, items.c.status)

chunks = Table(
    "vx_chunks",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("chunk_id", String(320), primary_key=True),
    Column("item_id", String(256), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("heading", Text),
    Column("token_count", Integer, nullable=False),
    Column("embedding", LargeBinary),
)
Index("ix_vx_chunks_item", chunks.c.tenant_id, chunks.c.item_id)

interactions = Table(
    "vx_interactions",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("event_id", String(256), primary_key=True),
    Column("user_id", String(256), nullable=False),
    Column("item_id", String(256), nullable=False),
    Column("type", String(32), nullable=False),
    Column("weight", Float, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("project_id", String(256)),
    Column("query", Text),
)
Index("ix_vx_inter_user_time", interactions.c.tenant_id, interactions.c.user_id, interactions.c.occurred_at)
Index("ix_vx_inter_item", interactions.c.tenant_id, interactions.c.item_id)
Index("ix_vx_inter_time", interactions.c.tenant_id, interactions.c.occurred_at)

processed_events = Table(
    "vx_processed_events",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("event_id", String(256), primary_key=True),
    Column("processed_at", DateTime(timezone=True), nullable=False),
)

jobs = Table(
    "vx_jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("status", String(16), nullable=False),
    Column("total", Integer, nullable=False, default=0),
    Column("processed", Integer, nullable=False, default=0),
    Column("failed", Integer, nullable=False, default=0),
    Column("errors", JSON, nullable=False, default=list),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("ix_vx_jobs_tenant", jobs.c.tenant_id, jobs.c.created_at)

tenant_state = Table(
    "vx_tenant_state",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("content_version", Integer, nullable=False, default=0),
    Column("model_trained_at", DateTime(timezone=True)),
)
