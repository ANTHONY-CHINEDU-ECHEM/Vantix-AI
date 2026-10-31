"""Ingestion pipeline: normalise, chunk, embed, persist and index items.

The pipeline is idempotent and change aware. Each item carries two fingerprints:
a content fingerprint (title and body) and a metadata fingerprint (status, ACL, project,
assignees and so on). Unchanged items are skipped entirely; items whose metadata alone
changed reuse their stored embeddings and only refresh index payloads, so permission
changes propagate in milliseconds without paying for re embedding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from vantix.domain.models import Chunk, Item
from vantix.embeddings.base import Embedder, l2_normalise
from vantix.index.base import LexicalIndex, VectorIndex
from vantix.ingestion.chunker import Chunker
from vantix.observability.logging import get_logger
from vantix.observability.metrics import CHUNKS_INDEXED, ITEMS_INDEXED, Timer
from vantix.storage.repository import Repository, decode_vector
from vantix.text import normalise

log = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    item_id: str
    status: str  # indexed, metadata_updated, unchanged, deleted, failed
    chunks: int = 0
    error: str | None = None


def fingerprints(item: Item) -> tuple[str, str]:
    content = hashlib.sha256(f"{item.title}\x1f{item.body}".encode()).hexdigest()
    meta_fields = item.model_dump(mode="json", exclude={"title", "body", "created_at", "updated_at"})
    meta = hashlib.sha256(json.dumps(meta_fields, sort_keys=True).encode()).hexdigest()
    return content[:32], meta[:32]


class IngestionService:
    def __init__(
        self,
        repository: Repository,
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        chunker: Chunker,
    ) -> None:
        self.repo = repository
        self.embedder = embedder
        self.vectors = vector_index
        self.lexical = lexical_index
        self.chunker = chunker

    # Public API -----------------------------------------------------------------------
    def ingest(self, tenant_id: str, items: list[Item]) -> list[IngestResult]:
        """Ingest a batch. Embeddings for every changed item are computed in one batched call."""
        if not items:
            return []
        # Last write wins inside a batch.
        unique = list({item.id: item for item in items}.values())
        existing = self.repo.get_items(tenant_id, [i.id for i in unique], include_deleted=True)

        results: dict[str, IngestResult] = {}
        to_embed: list[tuple[Item, list[Chunk], str]] = []
        to_refresh: list[tuple[Item, str]] = []

        for item in unique:
            item = item.model_copy(update={"title": normalise(item.title), "body": normalise(item.body)})
            content_fp, meta_fp = fingerprints(item)
            fingerprint = content_fp + meta_fp
            previous = existing.get(item.id)
            if previous is not None and not previous.deleted:
                if previous.content_hash == fingerprint:
                    results[item.id] = IngestResult(item.id, "unchanged")
                    continue
                if previous.content_hash[:32] == content_fp:
                    to_refresh.append((item, fingerprint))
                    continue
            to_embed.append((item, self._make_chunks(tenant_id, item), fingerprint))

        if to_embed:
            with Timer("ingest_embed"):
                self._embed_and_store(tenant_id, to_embed, results)
        for item, fingerprint in to_refresh:
            try:
                self._refresh_metadata(tenant_id, item, fingerprint)
                results[item.id] = IngestResult(item.id, "metadata_updated")
            except Exception as exc:  # noqa: BLE001
                log.exception("metadata_refresh_failed", extra={"item_id": item.id})
                results[item.id] = IngestResult(item.id, "failed", error=str(exc))

        if any(r.status in ("indexed", "metadata_updated") for r in results.values()):
            self.repo.bump_content_version(tenant_id)
        return [results[i.id] for i in unique]

    def delete(self, tenant_id: str, item_id: str) -> IngestResult:
        chunk_ids = self.repo.mark_deleted(tenant_id, item_id)
        self.vectors.delete(tenant_id, chunk_ids)
        self.lexical.delete(tenant_id, chunk_ids)
        self.repo.bump_content_version(tenant_id)
        return IngestResult(item_id, "deleted", chunks=len(chunk_ids))

    def reembed_all(self, tenant_id: str, batch_size: int = 64) -> int:
        """Rechunk and re embed every item. Required after changing embedding model or chunking.

        Returns the number of items processed. Items are processed in batches so memory stays flat
        regardless of knowledge base size; the service stays available throughout.
        """
        processed = 0
        batch: list[tuple[Item, list[Chunk], str]] = []
        results: dict[str, IngestResult] = {}
        for stored in self.repo.iter_items(tenant_id):
            item = stored.item
            content_fp, meta_fp = fingerprints(item)
            batch.append((item, self._make_chunks(tenant_id, item), content_fp + meta_fp))
            if len(batch) >= batch_size:
                self._embed_and_store(tenant_id, batch, results)
                processed += len(batch)
                batch = []
        if batch:
            self._embed_and_store(tenant_id, batch, results)
            processed += len(batch)
        self.repo.bump_content_version(tenant_id)
        return processed

    def rebuild_indexes(self, tenant_id: str, batch_size: int = 2000) -> int:
        """Rebuild derived indexes from the system of record. Returns the chunk count."""
        items = {s.item.id: s.item for s in self.repo.iter_items(tenant_id)}
        total = 0
        batch_chunks: list[Chunk] = []
        batch_vectors: list[np.ndarray] = []
        missing: list[Chunk] = []

        def flush() -> None:
            nonlocal total
            if batch_chunks:
                self.vectors.upsert(tenant_id, list(batch_chunks), np.vstack(batch_vectors))
                self.lexical.upsert(tenant_id, list(batch_chunks))
                total += len(batch_chunks)
                batch_chunks.clear()
                batch_vectors.clear()

        for row in self.repo.iter_chunk_rows(tenant_id):
            item = items.get(row["item_id"])
            if item is None:
                continue
            chunk = self._chunk_from_row(tenant_id, item, row)
            vector = decode_vector(row["embedding"])
            if vector is None or vector.shape[0] != self.embedder.dim:
                missing.append(chunk)
                continue
            batch_chunks.append(chunk)
            batch_vectors.append(vector)
            if len(batch_chunks) >= batch_size:
                flush()
        flush()
        if missing:
            log.warning("reembedding_chunks", extra={"tenant": tenant_id, "count": len(missing)})
            for start in range(0, len(missing), batch_size):
                part = missing[start : start + batch_size]
                vectors = self.embedder.embed_documents([c.text for c in part])
                self.vectors.upsert(tenant_id, part, vectors)
                self.lexical.upsert(tenant_id, part)
                total += len(part)
        return total

    # Internals ------------------------------------------------------------------------
    def _make_chunks(self, tenant_id: str, item: Item) -> list[Chunk]:
        pieces = self.chunker.chunk(item.title, item.body)
        if not pieces and item.title:
            from vantix.ingestion.chunker import TextChunk

            pieces = [TextChunk(0, item.title, None, 1)]
        return [self._chunk(tenant_id, item, p.ordinal, p.text, p.heading, p.token_count) for p in pieces]

    @staticmethod
    def _chunk(
        tenant_id: str, item: Item, ordinal: int, text: str, heading: str | None, token_count: int
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{item.id}::{ordinal}",
            item_id=item.id,
            tenant_id=tenant_id,
            ordinal=ordinal,
            text=text,
            heading=heading,
            token_count=token_count,
            kind=item.kind,
            project_id=item.project_id,
            visibility=item.acl.visibility,
            allowed_user_ids=item.acl.allowed_user_ids,
            allowed_group_ids=item.acl.allowed_group_ids,
            author_id=item.author_id,
            status=item.status,
            updated_at=item.updated_at,
        )

    def _chunk_from_row(self, tenant_id: str, item: Item, row: dict) -> Chunk:  # type: ignore[type-arg]
        return self._chunk(tenant_id, item, row["ordinal"], row["text"], row["heading"], row["token_count"])

    def _embed_and_store(
        self, tenant_id: str, work: list[tuple[Item, list[Chunk], str]], results: dict[str, IngestResult]
    ) -> None:
        texts = [chunk.text for _, chunks, _ in work for chunk in chunks]
        vectors = (
            self.embedder.embed_documents(texts) if texts else np.zeros((0, self.embedder.dim), np.float32)
        )
        offset = 0
        for item, chunks, fingerprint in work:
            item_vectors = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            try:
                old_chunk_ids = set(self.repo.chunk_ids_for_item(tenant_id, item.id))
                item_embedding = self._item_embedding(item_vectors)
                self.repo.upsert_item(
                    tenant_id,
                    item,
                    content_hash=fingerprint,
                    embedding=item_embedding,
                    chunks=list(zip(chunks, item_vectors, strict=True)),
                )
                stale = list(old_chunk_ids - {c.chunk_id for c in chunks})
                if stale:
                    self.vectors.delete(tenant_id, stale)
                    self.lexical.delete(tenant_id, stale)
                if chunks:
                    self.vectors.upsert(tenant_id, chunks, item_vectors)
                    self.lexical.upsert(tenant_id, chunks)
                ITEMS_INDEXED.labels(tenant=tenant_id).inc()
                CHUNKS_INDEXED.labels(tenant=tenant_id).inc(len(chunks))
                results[item.id] = IngestResult(item.id, "indexed", chunks=len(chunks))
            except Exception as exc:  # noqa: BLE001
                log.exception("ingest_item_failed", extra={"item_id": item.id})
                results[item.id] = IngestResult(item.id, "failed", error=str(exc))

    def _refresh_metadata(self, tenant_id: str, item: Item, fingerprint: str) -> None:
        self.repo.update_item_metadata_only(tenant_id, item, fingerprint)
        rows = [r for r in self._chunk_rows_for(tenant_id, item.id)]
        chunks = [self._chunk_from_row(tenant_id, item, r) for r in rows]
        vectors = [decode_vector(r["embedding"]) for r in rows]
        if chunks and all(v is not None for v in vectors):
            self.vectors.upsert(tenant_id, chunks, np.vstack(vectors))  # type: ignore[arg-type]
            self.lexical.upsert(tenant_id, chunks)

    def _chunk_rows_for(self, tenant_id: str, item_id: str) -> list[dict]:  # type: ignore[type-arg]
        from sqlalchemy import and_, select

        from vantix.storage import schema

        c = schema.chunks
        with self.repo.engine.connect() as conn:
            stmt = (
                select(c)
                .where(and_(c.c.tenant_id == tenant_id, c.c.item_id == item_id))
                .order_by(c.c.ordinal)
            )
            return [dict(r) for r in conn.execute(stmt).mappings()]

    @staticmethod
    def _item_embedding(vectors: np.ndarray) -> np.ndarray | None:
        if len(vectors) == 0:
            return None
        # Early chunks carry the title and introduction, so they are weighted more heavily.
        weights = 1.0 / np.sqrt(np.arange(1, len(vectors) + 1, dtype=np.float32))
        return l2_normalise((vectors * weights[:, None]).sum(axis=0))
