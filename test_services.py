from __future__ import annotations

from pathlib import Path

from tests.conftest import TENANT, make_settings
from vantix.container import Container
from vantix.domain.models import AccessControl, Citation, Item, ItemKind, Principal, Visibility
from vantix.evaluation.harness import evaluate_recommender, evaluate_retrieval
from vantix.index.base import SearchFilter
from vantix.rag.service import RagRequest, validate_citations
from vantix.recommender.engine import RecommendationContext, RecommendationKind, RecommendationQuery
from vantix.retrieval.hybrid import RetrievalRequest

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
ALICE = Principal(user_id="amara", project_ids=["p_platform"], group_ids=["eng", "leads"])
OUTSIDER = Principal(user_id="zoe", project_ids=[], group_ids=[])


def test_hybrid_retrieval_finds_relevant_items(seeded_container: Container) -> None:
    hits = seeded_container.retriever.retrieve(
        RetrievalRequest(TENANT, "how do I roll back a bad deployment", SearchFilter(principal=ALICE), k=5)
    )
    assert hits and hits[0].chunk.item_id == "kb_deploy_runbook"
    assert {"fused", "dense", "sparse", "relevance"} <= set(hits[0].signals)


def test_retrieval_enforces_acl(seeded_container: Container) -> None:
    hits = seeded_container.retriever.retrieve(
        RetrievalRequest(TENANT, "rollback deployment canary", SearchFilter(principal=OUTSIDER), k=20)
    )
    assert all(h.chunk.project_id is None or h.chunk.visibility is Visibility.TENANT for h in hits)
    salary = seeded_container.retriever.retrieve(
        RetrievalRequest(TENANT, "salary bands compensation", SearchFilter(principal=OUTSIDER), k=20)
    )
    assert "kb_salary_bands" not in {h.chunk.item_id for h in salary}
    lead = seeded_container.retriever.retrieve(
        RetrievalRequest(TENANT, "salary bands compensation", SearchFilter(principal=ALICE), k=20)
    )
    assert "kb_salary_bands" in {h.chunk.item_id for h in lead}


def test_rag_answer_is_cited_and_grounded(seeded_container: Container) -> None:
    answer = seeded_container.rag.answer(RagRequest(TENANT, ALICE, "How long does a rollback take?"))
    assert not answer.abstained
    assert answer.grounded and answer.citations
    assert any(c.item_id == "kb_deploy_runbook" for c in answer.citations)
    assert "[1]" in answer.answer or "[2]" in answer.answer


def test_rag_abstains_when_nothing_relevant(seeded_container: Container) -> None:
    answer = seeded_container.rag.answer(RagRequest(TENANT, ALICE, "zzqx vorpal flibbertigibbet"))
    assert answer.abstained and not answer.citations


def test_citation_validation_drops_unknown_indexes() -> None:
    src = [Citation(index=1, item_id="a", chunk_id="a::0", title="A", snippet="s", score=1.0)]
    text, used = validate_citations("Fact one [1]. Invented [7].", src)
    assert "[7]" not in text and [c.index for c in used] == [1]


def test_task_recommendations_respect_rules(seeded_container: Container) -> None:
    recs = seeded_container.recommender.recommend(
        RecommendationQuery(TENANT, ALICE, RecommendationKind.TASKS, k=10)
    )
    assert recs
    items = seeded_container.repository.get_items(TENANT, [r.target_id for r in recs])
    for rec in recs:
        item = items[rec.target_id].item
        assert item.kind is ItemKind.TASK
        assert item.status is not None and item.status.is_active
        assert not item.assignee_ids or "amara" in item.assignee_ids
        assert item.project_id in ALICE.project_ids
        assert rec.reasons, "every recommendation must be explained"


def test_document_recommendations_exclude_dismissed(seeded_container: Container) -> None:
    jack = Principal(user_id="jack", project_ids=["p_platform"], group_ids=["eng"])
    recs = seeded_container.recommender.recommend(
        RecommendationQuery(TENANT, jack, RecommendationKind.DOCUMENTS, k=20)
    )
    ids = [r.target_id for r in recs]
    assert "kb_feature_flags" not in ids
    assert all(i.startswith(("kb_", "dec_")) for i in ids)


def test_similar_and_contextual_recommendations(seeded_container: Container) -> None:
    similar = seeded_container.recommender.recommend(
        RecommendationQuery(
            TENANT,
            ALICE,
            RecommendationKind.SIMILAR,
            k=5,
            context=RecommendationContext(item_id="t_canary_alerts"),
            item_kinds=[ItemKind.TASK],
        )
    )
    assert similar and "t_canary_alerts" not in [r.target_id for r in similar]
    docs = seeded_container.recommender.recommend(
        RecommendationQuery(
            TENANT,
            ALICE,
            RecommendationKind.DOCUMENTS,
            k=5,
            context=RecommendationContext(item_id="t_canary_alerts"),
        )
    )
    assert any(r.target_id in {"kb_deploy_runbook", "kb_monitoring"} for r in docs)


def test_expert_recommendations(seeded_container: Container) -> None:
    experts = seeded_container.recommender.recommend(
        RecommendationQuery(
            TENANT,
            Principal(user_id="hugo", project_ids=["p_mobile"]),
            RecommendationKind.EXPERTS,
            k=3,
            context=RecommendationContext(item_id="t_refresh_token_bug"),
        )
    )
    assert experts
    assert experts[0].target_type == "user"
    assert {e.target_id for e in experts} & {"dev", "grace", "chloe"}
    assert "hugo" not in {e.target_id for e in experts}


def test_assist_combines_everything(seeded_container: Container) -> None:
    from vantix.assist import AssistRequest

    result = seeded_container.assist.assist(
        AssistRequest(
            TENANT, ALICE, item_id="t_canary_alerts", question="When should I stop a canary rollout?"
        )
    )
    assert result.answer is not None and result.answer.citations
    assert result.documents and result.experts is not None


def test_ingestion_is_change_aware(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    item = Item(id="doc", title="Title", body="Body text about releases", project_id="p")
    assert c.ingestion.ingest("t", [item])[0].status == "indexed"
    assert c.ingestion.ingest("t", [item])[0].status == "unchanged"
    moved = item.model_copy(
        update={"acl": AccessControl(visibility=Visibility.RESTRICTED, allowed_user_ids=["x"])}
    )
    assert c.ingestion.ingest("t", [moved])[0].status == "metadata_updated"
    hits = c.vector_index.search(
        "t",
        c.embedder.embed_query("releases"),
        5,
        SearchFilter(principal=Principal(user_id="y", project_ids=["p"])),
    )
    assert hits == [], "ACL change must propagate to the index without re embedding"
    assert c.ingestion.delete("t", "doc").status == "deleted"
    assert c.vector_index.count("t") == 0 and c.lexical_index.count("t") == 0
    c.stop()


def test_indexes_rebuild_from_system_of_record(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = Container.build(settings)
    first.ingestion.ingest("t", [Item(id="a", title="Kafka retention", body="Seven days on the cluster.")])
    first.stop()
    second = Container.build(settings)
    second.start()
    assert second.vector_index.count("t") == 1 and second.lexical_index.count("t") == 1
    second.stop()


def test_evaluation_harness(seeded_container: Container) -> None:
    retrieval = evaluate_retrieval(seeded_container, TENANT, DATA / "eval_queries.jsonl", k=10)
    assert retrieval["metrics"]["hit_rate@10"] >= 0.8
    rec = evaluate_recommender(seeded_container, TENANT, k=10)
    assert rec["users_evaluated"] > 0 and "ndcg@10" in rec["metrics"]


def test_model_persists_and_reloads(seeded_container: Container) -> None:
    from vantix.recommender.model import ModelRegistry

    reloaded = ModelRegistry(seeded_container.settings.model_dir).load(TENANT)
    assert reloaded is not None and reloaded.als is not None and reloaded.graph is not None
    assert reloaded.item_ids == seeded_container.registry.get(TENANT).item_ids


def test_full_reembed_rebuilds_everything(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    c.ingestion.ingest("t", [Item(id="a", title="Alpha", body="Rotate secrets every 90 days.")])
    assert c.ingestion.reembed_all("t") == 1
    assert c.vector_index.count("t") == 1 and c.lexical_index.count("t") == 1
    c.stop()
