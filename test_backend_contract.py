"""Contract tests every vector and lexical index must satisfy.

Memory backends always run. Qdrant and OpenSearch run when ``VANTIX_TEST_QDRANT_URL`` and
``VANTIX_TEST_OPENSEARCH_URL`` are set, which the CI integration job does with service
containers. The same assertions therefore guard every backend, including access control.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import httpx
import pytest

from tests.conftest import OPENSEARCH_URL, QDRANT_URL
from vantix.domain.models import Chunk, ItemKind, Principal, TaskStatus, Visibility
from vantix.embeddings.hashing import HashingEmbedder
from vantix.index.base import LexicalIndex, SearchFilter, VectorIndex
from vantix.index.memory import MemoryBM25Index, MemoryVectorIndex

DIM = 64
EMBED = HashingEmbedder(DIM)

CHUNKS = [
    Chunk(
        chunk_id="pub::0",
        item_id="pub",
        tenant_id="t1",
        ordinal=0,
        text="rollback the release pipeline",
        visibility=Visibility.TENANT,
    ),
    Chunk(
        chunk_id="proj::0",
        item_id="proj",
        tenant_id="t1",
        ordinal=0,
        text="rollback steps for project alpha",
        project_id="alpha",
    ),
    Chunk(
        chunk_id="secret::0",
        item_id="secret",
        tenant_id="t1",
        ordinal=0,
        text="rollback secret salary data",
        visibility=Visibility.RESTRICTED,
        allowed_group_ids=["leads"],
    ),
    Chunk(
        chunk_id="task::0",
        item_id="task",
        tenant_id="t1",
        ordinal=0,
        text="rollback task already done",
        kind=ItemKind.TASK,
        status=TaskStatus.DONE,
        visibility=Visibility.TENANT,
    ),
]
OTHER_TENANT = Chunk(
    chunk_id="pub::0",
    item_id="pub",
    tenant_id="t2",
    ordinal=0,
    text="rollback elsewhere",
    visibility=Visibility.TENANT,
)


def _vector_backends() -> list[str]:
    return ["memory"] + (["qdrant"] if QDRANT_URL else [])


def _lexical_backends() -> list[str]:
    return ["memory"] + (["opensearch"] if OPENSEARCH_URL else [])


@pytest.fixture(params=_vector_backends())
def vector_index(request: pytest.FixtureRequest) -> Iterator[VectorIndex]:
    if request.param == "memory":
        yield MemoryVectorIndex(DIM)
        return
    from vantix.index.qdrant import QdrantVectorIndex

    name = f"contract_{uuid.uuid4().hex[:8]}"
    index = QdrantVectorIndex(QDRANT_URL or "", name, DIM)
    yield index
    httpx.delete(f"{QDRANT_URL}/collections/{name}")
    index.close()


@pytest.fixture(params=_lexical_backends())
def lexical_index(request: pytest.FixtureRequest) -> Iterator[LexicalIndex]:
    if request.param == "memory":
        yield MemoryBM25Index()
        return
    from vantix.index.opensearch import OpenSearchLexicalIndex

    name = f"contract_{uuid.uuid4().hex[:8]}"
    index = OpenSearchLexicalIndex(OPENSEARCH_URL or "", name)
    yield index
    httpx.delete(f"{OPENSEARCH_URL}/{name}")
    index.close()


def _ids(hits) -> set[str]:  # type: ignore[no-untyped-def]
    return {h.chunk.item_id for h in hits}


def test_vector_contract(vector_index: VectorIndex) -> None:
    vector_index.upsert("t1", CHUNKS, EMBED.embed_documents([c.text for c in CHUNKS]))
    vector_index.upsert("t2", [OTHER_TENANT], EMBED.embed_documents([OTHER_TENANT.text]))
    query = EMBED.embed_query("rollback")
    member = SearchFilter(principal=Principal(user_id="u", project_ids=["alpha"]))
    outsider = SearchFilter(principal=Principal(user_id="x"))
    lead = SearchFilter(principal=Principal(user_id="l", group_ids=["leads"]))

    assert _ids(vector_index.search("t1", query, 10, member)) == {"pub", "proj", "task"}
    assert _ids(vector_index.search("t1", query, 10, outsider)) == {"pub", "task"}
    assert "secret" in _ids(vector_index.search("t1", query, 10, lead))
    assert _ids(
        vector_index.search("t1", query, 10, SearchFilter(principal=Principal(user_id="a", is_admin=True)))
    ) == {"pub", "proj", "secret", "task"}
    assert "task" not in _ids(vector_index.search("t1", query, 10, SearchFilter(active_tasks_only=True)))
    assert _ids(vector_index.search("t1", query, 10, SearchFilter(kinds=[ItemKind.TASK]))) == {"task"}
    assert "pub" not in _ids(vector_index.search("t1", query, 10, SearchFilter(exclude_item_ids={"pub"})))
    assert _ids(vector_index.search("t2", query, 10, SearchFilter())) == {"pub"}, "tenants are isolated"
    assert vector_index.count("t1") == 4

    vector_index.delete("t1", ["pub::0"])
    assert vector_index.count("t1") == 3
    assert vector_index.count("t2") == 1, "deleting in one tenant never touches another"
    vector_index.drop_tenant("t1")
    assert vector_index.count("t1") == 0
    assert vector_index.healthy()


def test_lexical_contract(lexical_index: LexicalIndex) -> None:
    lexical_index.upsert("t1", CHUNKS)
    lexical_index.upsert("t2", [OTHER_TENANT])
    member = SearchFilter(principal=Principal(user_id="u", project_ids=["alpha"]))
    outsider = SearchFilter(principal=Principal(user_id="x"))

    assert _ids(lexical_index.search("t1", "rollback", 10, member)) == {"pub", "proj", "task"}
    assert _ids(lexical_index.search("t1", "rollback", 10, outsider)) == {"pub", "task"}
    assert _ids(lexical_index.search("t1", "salary", 10, outsider)) == set()
    top = lexical_index.search("t1", "alpha", 10, member)
    assert top and top[0].chunk.item_id == "proj"
    assert _ids(lexical_index.search("t2", "rollback", 10, SearchFilter())) == {"pub"}
    assert lexical_index.count("t1") == 4

    lexical_index.delete("t1", ["proj::0"])
    assert "proj" not in _ids(lexical_index.search("t1", "alpha", 10, member))
    lexical_index.drop_tenant("t1")
    assert lexical_index.count("t1") == 0
    assert lexical_index.count("t2") == 1
    assert lexical_index.healthy()


def test_opensearch_adapter_request_shapes() -> None:
    """Exercise the OpenSearch adapter against a recording fake (runs everywhere)."""
    from vantix.index.opensearch import OpenSearchLexicalIndex

    requests: list[tuple[str, str, bytes]] = []
    stored: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        if request.method == "HEAD":
            return httpx.Response(404)
        if request.method == "PUT":
            return httpx.Response(200, json={"acknowledged": True})
        if request.url.path == "/_bulk":
            lines = request.content.decode().strip().split("\n")
            items = []
            i = 0
            while i < len(lines):
                action = json.loads(lines[i])
                op, meta = next(iter(action.items()))
                if op == "index":
                    stored[meta["_id"]] = json.loads(lines[i + 1])
                    i += 2
                else:
                    stored.pop(meta["_id"], None)
                    i += 1
                items.append({op: {"status": 200}})
            return httpx.Response(200, json={"errors": False, "items": items})
        if request.url.path.endswith("/_search"):
            hits = [{"_source": src, "_score": 1.0} for src in stored.values()]
            return httpx.Response(200, json={"hits": {"hits": hits}})
        if request.url.path.endswith("/_count"):
            return httpx.Response(200, json={"count": len(stored)})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx.Client, "__init__", patched)
        index = OpenSearchLexicalIndex("https://search.test", "vx")
        index.upsert("t1", CHUNKS)
        hits = index.search("t1", "rollback", 10, SearchFilter(principal=Principal(user_id="x")))
        index.delete("t1", ["pub::0"])
        assert index.count("t1") == 3

    assert ("PUT", "/vx") in [(m, p) for m, p, _ in requests], "index is created with a mapping"
    mapping = json.loads(next(c for m, p, c in requests if m == "PUT"))
    assert mapping["mappings"]["dynamic"] == "strict"
    assert _ids(hits) == {"pub", "task"}, "in memory recheck enforces ACL even if the backend over returns"
    search_body = json.loads(next(c for m, p, c in requests if p.endswith("/_search")))
    filters = search_body["query"]["bool"]["filter"]
    assert filters[0] == {"term": {"tenant_id": "t1"}}


def test_opensearch_bulk_errors_surface() -> None:
    from vantix.index.opensearch import OpenSearchLexicalIndex

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        return httpx.Response(200, json={"errors": True, "items": [{"index": {"error": {"type": "mapper"}}}]})

    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx.Client, "__init__", patched)
        index = OpenSearchLexicalIndex("https://search.test", "vx")
        with pytest.raises(RuntimeError, match="1 failures"):
            index.upsert("t1", CHUNKS[:1])


def _patch_transport(mp: pytest.MonkeyPatch, handler) -> None:  # type: ignore[no-untyped-def]
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    mp.setattr(httpx.Client, "__init__", patched)


def test_qdrant_adapter_request_shapes() -> None:
    """Exercise the Qdrant adapter against a recording fake (runs everywhere)."""
    from vantix.index.qdrant import QdrantVectorIndex, point_id

    calls: list[tuple[str, str, dict]] = []
    points: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, request.url.path, body))
        path = request.url.path
        if request.method == "GET" and path == "/collections/vx":
            return httpx.Response(404)
        if path.endswith("/points") and request.method == "PUT":
            for p in body["points"]:
                points[p["id"]] = p
            return httpx.Response(200, json={"result": {}})
        if path.endswith("/points/search"):
            hits = [{"payload": p["payload"], "score": 0.5} for p in points.values()]
            return httpx.Response(200, json={"result": hits})
        if path.endswith("/points/delete"):
            for pid in body.get("points", []):
                points.pop(pid, None)
            if "filter" in body:
                points.clear()
            return httpx.Response(200, json={"result": {}})
        if path.endswith("/points/count"):
            return httpx.Response(200, json={"result": {"count": len(points)}})
        if path == "/readyz":
            return httpx.Response(200)
        return httpx.Response(200, json={"result": {}})

    with pytest.MonkeyPatch.context() as mp:
        _patch_transport(mp, handler)
        index = QdrantVectorIndex("https://qdrant.test", "vx", DIM, api_key="k")
        index.upsert("t1", CHUNKS, EMBED.embed_documents([c.text for c in CHUNKS]))
        hits = index.search(
            "t1", EMBED.embed_query("rollback"), 10, SearchFilter(principal=Principal(user_id="x"))
        )
        assert _ids(hits) == {"pub", "task"}, (
            "in memory recheck enforces ACL even if the backend over returns"
        )
        index.delete("t1", ["pub::0"])
        index.delete("t1", [])
        assert index.count("t1") == 3
        index.drop_tenant("t1")
        assert index.count("t1") == 0 and index.healthy()

    created = [c for c in calls if c[0] == "PUT" and c[1] == "/collections/vx"]
    assert created and created[0][2]["vectors"]["size"] == DIM
    indexes = [c[2]["field_name"] for c in calls if c[1] == "/collections/vx/index"]
    assert indexes[0] == "tenant_id" and {"allowed_user_ids", "project_id", "updated_at"} <= set(indexes)
    search = next(c[2] for c in calls if c[1].endswith("/points/search"))
    assert search["filter"]["must"][0] == {"key": "tenant_id", "match": {"value": "t1"}}
    assert point_id("t1", "a::0") == point_id("t1", "a::0") != point_id("t2", "a::0")


def test_qdrant_rejects_dimension_mismatch() -> None:
    from vantix.index.qdrant import QdrantVectorIndex

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"config": {"params": {"vectors": {"size": 999}}}}})

    with pytest.MonkeyPatch.context() as mp:
        _patch_transport(mp, handler)
        with pytest.raises(RuntimeError, match="dimension"):
            QdrantVectorIndex("https://qdrant.test", "vx", DIM)
