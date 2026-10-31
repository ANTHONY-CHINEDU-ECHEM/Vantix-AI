"""Qdrant vector index adapter (REST API, no client library required).

A single collection holds every tenant. Tenancy and access control are enforced with
indexed payload filters, which Qdrant applies during HNSW traversal, so filtered searches
stay fast and never leak results across tenants. ``tenant_id`` is created as a tenant
aware payload index so Qdrant can co locate each tenant's vectors.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import numpy as np

from vantix.domain.models import Chunk
from vantix.http import request_with_retries
from vantix.index.base import IndexHit, SearchFilter, VectorIndex
from vantix.observability.logging import get_logger

log = get_logger(__name__)

_NAMESPACE = uuid.UUID("6f1c2a52-1d8e-4a51-9a3b-6e0d1f6b7c11")
_KEYWORD_FIELDS = (
    "item_id",
    "chunk_id",
    "kind",
    "project_id",
    "visibility",
    "allowed_user_ids",
    "allowed_group_ids",
    "author_id",
    "status",
)


def point_id(tenant_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{tenant_id}\x1f{chunk_id}"))


class QdrantVectorIndex(VectorIndex):
    def __init__(self, url: str, collection: str, dim: int, api_key: str = "", hnsw_ef: int = 128) -> None:
        headers = {"api-key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30.0)
        self.collection = collection
        self.dim = dim
        self.hnsw_ef = hnsw_ef
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        response = self._client.get(f"/collections/{self.collection}")
        if response.status_code == 200:
            size = response.json()["result"]["config"]["params"]["vectors"]["size"]
            if int(size) != self.dim:
                raise RuntimeError(
                    f"Qdrant collection {self.collection} has dimension {size}, expected {self.dim}"
                )
            return
        request_with_retries(
            self._client,
            "PUT",
            f"/collections/{self.collection}",
            json={
                "vectors": {"size": self.dim, "distance": "Cosine", "on_disk": True},
                "hnsw_config": {"m": 16, "ef_construct": 200, "payload_m": 16},
                "optimizers_config": {"default_segment_number": 4},
                "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}},
                "on_disk_payload": True,
            },
        )
        request_with_retries(
            self._client,
            "PUT",
            f"/collections/{self.collection}/index",
            json={"field_name": "tenant_id", "field_schema": {"type": "keyword", "is_tenant": True}},
        )
        for name in _KEYWORD_FIELDS:
            request_with_retries(
                self._client,
                "PUT",
                f"/collections/{self.collection}/index",
                json={"field_name": name, "field_schema": "keyword"},
            )
        request_with_retries(
            self._client,
            "PUT",
            f"/collections/{self.collection}/index",
            json={"field_name": "updated_at", "field_schema": "datetime"},
        )
        log.info("qdrant_collection_created", extra={"collection": self.collection})

    def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for start in range(0, len(chunks), 256):
            batch = chunks[start : start + 256]
            points = [
                {
                    "id": point_id(tenant_id, chunk.chunk_id),
                    "vector": vectors[start + i].astype(float).tolist(),
                    "payload": chunk.model_dump(mode="json"),
                }
                for i, chunk in enumerate(batch)
            ]
            request_with_retries(
                self._client,
                "PUT",
                f"/collections/{self.collection}/points?wait=true",
                json={"points": points},
            )

    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        request_with_retries(
            self._client,
            "POST",
            f"/collections/{self.collection}/points/delete?wait=true",
            json={"points": [point_id(tenant_id, cid) for cid in chunk_ids]},
        )

    def drop_tenant(self, tenant_id: str) -> None:
        request_with_retries(
            self._client,
            "POST",
            f"/collections/{self.collection}/points/delete?wait=true",
            json={"filter": {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]}},
        )

    def search(self, tenant_id: str, vector: np.ndarray, k: int, flt: SearchFilter) -> list[IndexHit]:
        body = {
            "vector": np.asarray(vector, dtype=float).tolist(),
            "filter": build_qdrant_filter(tenant_id, flt),
            "limit": k,
            "with_payload": True,
            "params": {"hnsw_ef": max(self.hnsw_ef, k), "quantization": {"rescore": True}},
        }
        response = request_with_retries(
            self._client, "POST", f"/collections/{self.collection}/points/search", json=body
        )
        hits: list[IndexHit] = []
        for point in response.json().get("result", []):
            chunk = Chunk.model_validate(point["payload"])
            if flt.matches(chunk):  # defence in depth
                hits.append(IndexHit(chunk, float(point["score"])))
        return hits

    def count(self, tenant_id: str) -> int:
        response = request_with_retries(
            self._client,
            "POST",
            f"/collections/{self.collection}/points/count",
            json={"filter": {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]}, "exact": True},
        )
        return int(response.json()["result"]["count"])

    def healthy(self) -> bool:
        try:
            return self._client.get("/readyz").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


def build_qdrant_filter(tenant_id: str, flt: SearchFilter) -> dict[str, Any]:
    must: list[dict[str, Any]] = [{"key": "tenant_id", "match": {"value": tenant_id}}]
    must_not: list[dict[str, Any]] = []
    if flt.kinds:
        must.append({"key": "kind", "match": {"any": [k.value for k in flt.kinds]}})
    if flt.project_ids:
        must.append({"key": "project_id", "match": {"any": list(flt.project_ids)}})
    if flt.item_ids is not None:
        must.append({"key": "item_id", "match": {"any": list(flt.item_ids) or ["\x00"]}})
    if flt.exclude_item_ids:
        must_not.append({"key": "item_id", "match": {"any": list(flt.exclude_item_ids)}})
    if flt.active_tasks_only:
        must_not.append({"key": "status", "match": {"any": ["done", "cancelled"]}})
    if flt.updated_after is not None:
        must.append({"key": "updated_at", "range": {"gte": flt.updated_after.isoformat()}})
    principal = flt.principal
    if principal is not None and not principal.is_admin:
        should: list[dict[str, Any]] = [
            {"key": "visibility", "match": {"value": "tenant"}},
            {"key": "allowed_user_ids", "match": {"value": principal.user_id}},
            {"key": "author_id", "match": {"value": principal.user_id}},
            {
                "must": [
                    {"key": "visibility", "match": {"value": "project"}},
                    {"is_empty": {"key": "project_id"}},
                ]
            },
        ]
        if principal.group_ids:
            should.append({"key": "allowed_group_ids", "match": {"any": list(principal.group_ids)}})
        if principal.project_ids:
            should.append(
                {
                    "must": [
                        {"key": "visibility", "match": {"value": "project"}},
                        {"key": "project_id", "match": {"any": list(principal.project_ids)}},
                    ]
                }
            )
        must.append({"should": should})
    result: dict[str, Any] = {"must": must}
    if must_not:
        result["must_not"] = must_not
    return result
