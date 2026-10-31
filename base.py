"""Index contracts and the shared search filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from vantix.domain.models import Chunk, ItemKind, Principal, TaskStatus
from vantix.security.acl import can_view_chunk


@dataclass(slots=True)
class SearchFilter:
    """Constraints applied inside the index query and re checked in memory."""

    principal: Principal | None = None
    kinds: list[ItemKind] | None = None
    project_ids: list[str] | None = None
    item_ids: list[str] | None = None
    exclude_item_ids: set[str] = field(default_factory=set)
    active_tasks_only: bool = False
    updated_after: datetime | None = None

    def matches(self, chunk: Chunk) -> bool:
        if self.principal is not None and not can_view_chunk(self.principal, chunk):
            return False
        if self.kinds and chunk.kind not in self.kinds:
            return False
        if self.project_ids and chunk.project_id not in self.project_ids:
            return False
        if self.item_ids is not None and chunk.item_id not in self.item_ids:
            return False
        if chunk.item_id in self.exclude_item_ids:
            return False
        if (
            self.active_tasks_only
            and chunk.kind is ItemKind.TASK
            and (chunk.status is None or not TaskStatus(chunk.status).is_active)
        ):
            return False
        return not (self.updated_after is not None and chunk.updated_at < self.updated_after)


@dataclass(slots=True)
class IndexHit:
    chunk: Chunk
    score: float


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def search(self, tenant_id: str, vector: np.ndarray, k: int, flt: SearchFilter) -> list[IndexHit]: ...

    @abstractmethod
    def count(self, tenant_id: str) -> int: ...

    def drop_tenant(self, tenant_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def healthy(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover
        return None


class LexicalIndex(ABC):
    @abstractmethod
    def upsert(self, tenant_id: str, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def search(self, tenant_id: str, query: str, k: int, flt: SearchFilter) -> list[IndexHit]: ...

    @abstractmethod
    def count(self, tenant_id: str) -> int: ...

    def drop_tenant(self, tenant_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def healthy(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover
        return None
