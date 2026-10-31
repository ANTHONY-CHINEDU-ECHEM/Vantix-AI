"""OpenSearch (or Elasticsearch 8 compatible) lexical index adapter.

Uses the English analyser for BM25 on chunk text, a boosted title subfield, and keyword
fields for tenant isolation and access control filters executed inside the query.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from vantix.domain.models import Chunk
from vantix.http import request_with_retries
from vantix.index.base import IndexHit, LexicalIndex, SearchFilter

_MAPPING = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "vantix_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "english_stop", "english_stemmer"],
                }
            },
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "light_english"},
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "tenant_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "item_id": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "text": {"type": "text", "analyzer": "vantix_text"},
            "heading": {"type": "text", "analyzer": "vantix_text"},
            "token_count": {"type": "integer"},
            "kind": {"type": "keyword"},
            "project_id": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "allowed_user_ids": {"type": "keyword"},
            "allowed_group_ids": {"type": "keyword"},
            "author_id": {"type": "keyword"},
            "status": {"type": "keyword"},
            "updated_at": {"type": "date"},
        },
    },
}


class OpenSearchLexicalIndex(LexicalIndex):
    def __init__(self, url: str, index: str, username: str = "", password: str = "") -> None:  # nosec B107
        auth = (username, password) if username else None
        self._client = httpx.Client(base_url=url.rstrip("/"), auth=auth, timeout=30.0)
        self.index = index
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self._client.head(f"/{self.index}").status_code == 200:
            return
        request_with_retries(self._client, "PUT", f"/{self.index}", json=_MAPPING)

    @staticmethod
    def _doc_id(tenant_id: str, chunk_id: str) -> str:
        return f"{tenant_id}:{chunk_id}"

    def upsert(self, tenant_id: str, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), 500):
            lines: list[str] = []
            for chunk in chunks[start : start + 500]:
                lines.append(
                    json.dumps(
                        {"index": {"_index": self.index, "_id": self._doc_id(tenant_id, chunk.chunk_id)}}
                    )
                )
                lines.append(json.dumps(chunk.model_dump(mode="json")))
            self._bulk(lines)

    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        lines = [
            json.dumps({"delete": {"_index": self.index, "_id": self._doc_id(tenant_id, cid)}})
            for cid in chunk_ids
        ]
        self._bulk(lines)

    def _bulk(self, lines: list[str]) -> None:
        response = request_with_retries(
            self._client,
            "POST",
            "/_bulk?refresh=wait_for",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        body = response.json()
        if body.get("errors"):
            failures = [i for i in body.get("items", []) if next(iter(i.values())).get("error")]
            raise RuntimeError(f"OpenSearch bulk request had {len(failures)} failures: {failures[:3]}")

    def drop_tenant(self, tenant_id: str) -> None:
        request_with_retries(
            self._client,
            "POST",
            f"/{self.index}/_delete_by_query?refresh=true",
            json={"query": {"term": {"tenant_id": tenant_id}}},
        )

    def search(self, tenant_id: str, query: str, k: int, flt: SearchFilter) -> list[IndexHit]:
        body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text", "heading^2"],
                                "type": "best_fields",
                                "operator": "or",
                            }
                        }
                    ],
                    "filter": build_opensearch_filter(tenant_id, flt),
                    "must_not": _must_not(flt),
                }
            },
        }
        response = request_with_retries(self._client, "POST", f"/{self.index}/_search", json=body)
        hits: list[IndexHit] = []
        for hit in response.json()["hits"]["hits"]:
            chunk = Chunk.model_validate(hit["_source"])
            if flt.matches(chunk):
                hits.append(IndexHit(chunk, float(hit["_score"])))
        return hits

    def count(self, tenant_id: str) -> int:
        response = request_with_retries(
            self._client, "POST", f"/{self.index}/_count", json={"query": {"term": {"tenant_id": tenant_id}}}
        )
        return int(response.json()["count"])

    def healthy(self) -> bool:
        try:
            return self._client.get("/_cluster/health").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


def _must_not(flt: SearchFilter) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    if flt.exclude_item_ids:
        clauses.append({"terms": {"item_id": list(flt.exclude_item_ids)}})
    if flt.active_tasks_only:
        clauses.append({"terms": {"status": ["done", "cancelled"]}})
    return clauses


def build_opensearch_filter(tenant_id: str, flt: SearchFilter) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [{"term": {"tenant_id": tenant_id}}]
    if flt.kinds:
        clauses.append({"terms": {"kind": [k.value for k in flt.kinds]}})
    if flt.project_ids:
        clauses.append({"terms": {"project_id": list(flt.project_ids)}})
    if flt.item_ids is not None:
        clauses.append({"terms": {"item_id": list(flt.item_ids)}})
    if flt.updated_after is not None:
        clauses.append({"range": {"updated_at": {"gte": flt.updated_after.isoformat()}}})
    principal = flt.principal
    if principal is not None and not principal.is_admin:
        should: list[dict[str, Any]] = [
            {"term": {"visibility": "tenant"}},
            {"term": {"allowed_user_ids": principal.user_id}},
            {"term": {"author_id": principal.user_id}},
            {
                "bool": {
                    "filter": [{"term": {"visibility": "project"}}],
                    "must_not": [{"exists": {"field": "project_id"}}],
                }
            },
        ]
        if principal.group_ids:
            should.append({"terms": {"allowed_group_ids": list(principal.group_ids)}})
        if principal.project_ids:
            should.append(
                {
                    "bool": {
                        "filter": [
                            {"term": {"visibility": "project"}},
                            {"terms": {"project_id": list(principal.project_ids)}},
                        ]
                    }
                }
            )
        clauses.append({"bool": {"should": should, "minimum_should_match": 1}})
    return clauses
