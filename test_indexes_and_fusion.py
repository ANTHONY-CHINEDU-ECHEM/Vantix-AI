import numpy as np

from vantix.domain.models import Chunk, ItemKind, Principal, TaskStatus, Visibility
from vantix.embeddings.hashing import HashingEmbedder
from vantix.index.base import SearchFilter
from vantix.index.memory import MemoryBM25Index, MemoryVectorIndex
from vantix.index.opensearch import build_opensearch_filter
from vantix.index.qdrant import build_qdrant_filter
from vantix.retrieval.fusion import jaccard, min_max, mmr, weighted_rrf
from vantix.retrieval.rerankers import LexicalReranker


def chunk(cid: str, text: str, **kw: object) -> Chunk:
    return Chunk(chunk_id=cid, item_id=cid.split("::")[0], tenant_id="t", ordinal=0, text=text, **kw)  # type: ignore[arg-type]


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(128)
    a = e.embed_documents(["database migration index"])[0]
    b = e.embed_query("database migration index")
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5
    near = e.embed_query("index migration on a database table")
    far = e.embed_query("mobile push notification quiet hours")
    assert a @ near > a @ far


def test_memory_vector_index_upsert_delete_and_filter() -> None:
    e = HashingEmbedder(64)
    idx = MemoryVectorIndex(64)
    chunks = [
        chunk("a::0", "rollback the deployment", project_id="p1"),
        chunk("b::0", "rollback secret", visibility=Visibility.RESTRICTED, allowed_user_ids=["bob"]),
        chunk("c::0", "deployment canary", kind=ItemKind.TASK, status=TaskStatus.DONE),
    ]
    idx.upsert("t", chunks, e.embed_documents([c.text for c in chunks]))
    q = e.embed_query("rollback")
    alice = SearchFilter(principal=Principal(user_id="alice", project_ids=["p1"]))
    ids = [h.chunk.item_id for h in idx.search("t", q, 10, alice)]
    assert "b" not in ids and "a" in ids
    bob = SearchFilter(principal=Principal(user_id="bob"))
    assert "b" in [h.chunk.item_id for h in idx.search("t", q, 10, bob)]
    assert idx.search("other", q, 10, alice) == []
    active = SearchFilter(active_tasks_only=True)
    assert "c" not in [h.chunk.item_id for h in idx.search("t", q, 10, active)]
    idx.delete("t", ["a::0"])
    assert idx.count("t") == 2
    idx.upsert("t", [chunks[0]], e.embed_documents([chunks[0].text]))
    assert idx.count("t") == 3


def test_bm25_ranks_exact_terms_and_supports_updates() -> None:
    idx = MemoryBM25Index()
    idx.upsert("t", [chunk("a::0", "error code E4012 when syncing"), chunk("b::0", "syncing works fine")])
    hits = idx.search("t", "E4012", 5, SearchFilter())
    assert [h.chunk.item_id for h in hits] == ["a"]
    idx.upsert("t", [chunk("a::0", "nothing relevant here")])
    assert idx.search("t", "E4012", 5, SearchFilter()) == []
    idx.delete("t", ["b::0"])
    assert idx.count("t") == 1


def test_weighted_rrf_and_helpers() -> None:
    fused = weighted_rrf({"d": ["x", "y"], "s": ["y", "z"]}, {"d": 1.0, "s": 1.0}, k=60)
    assert max(fused, key=fused.__getitem__) == "y"
    assert min_max({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    picked = mmr(
        ["a", "a2", "b"],
        relevance={"a": 1.0, "a2": 0.99, "b": 0.8}.__getitem__,
        similarity=lambda x, y: 1.0 if x[0] == y[0] else 0.0,
        k=2,
        lambda_=0.5,
    )
    assert picked == ["a", "b"]


def test_lexical_reranker_prefers_phrase_matches() -> None:
    scores = LexicalReranker().score(
        "refresh token reuse",
        ["refresh token reuse revokes the session", "the token was reused by refresh logic"],
    )
    assert scores[0] > scores[1]


def test_external_filters_encode_acl() -> None:
    flt = SearchFilter(
        principal=Principal(user_id="u", group_ids=["g"], project_ids=["p"]),
        kinds=[ItemKind.TASK],
        exclude_item_ids={"x"},
        active_tasks_only=True,
    )
    q = build_qdrant_filter("t", flt)
    assert q["must"][0] == {"key": "tenant_id", "match": {"value": "t"}}
    assert any("should" in m for m in q["must"])
    assert q["must_not"]
    os_filter = build_opensearch_filter("t", flt)
    assert os_filter[0] == {"term": {"tenant_id": "t"}}
    assert any("bool" in clause for clause in os_filter)
    admin = build_qdrant_filter("t", SearchFilter(principal=Principal(user_id="a", is_admin=True)))
    assert not any("should" in m for m in admin["must"])
