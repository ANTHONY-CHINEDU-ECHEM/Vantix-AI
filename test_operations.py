"""Regression tests for audit fixes, plus the CLI, Python client, scheduler, cache and logging."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import API_KEY, DATA, SECRET, make_settings, wait_for
from vantix.cache import TTLCache
from vantix.client import VantixClient, VantixClientError
from vantix.container import Container
from vantix.domain.models import Interaction, InteractionType, Item
from vantix.errors import QueueFull
from vantix.events.schemas import EventBatch
from vantix.observability.logging import JsonFormatter, request_id_var, tenant_id_var
from vantix.recommender.als import ALSConfig, ImplicitALS
from vantix.security.auth import ApiKeyAuthenticator, RateLimiter, sign_payload, verify_signature

PRINCIPAL = {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng"]}


# Audit fixes ---------------------------------------------------------------------------
def test_queue_full_releases_claims_so_retry_is_not_lost(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    batch = EventBatch.model_validate(
        {
            "events": [
                {"event_id": "e1", "type": "item.upserted", "data": {"id": "doc", "title": "T", "body": "B"}}
            ]
        }
    )

    def full(*_args: object, **_kwargs: object) -> str:
        raise QueueFull("full")

    original = c.queue.submit
    c.queue.submit = full  # type: ignore[method-assign]
    with pytest.raises(QueueFull):
        c.events.process("t", batch)
    c.queue.submit = original  # type: ignore[method-assign]
    c.queue.start()
    result = c.events.process("t", batch)
    assert result.duplicates == 0 and result.items_queued == 1, "the retried event must be processed"
    c.queue.drain()
    assert c.repository.get_item("t", "doc") is not None
    c.stop()


def test_interactions_bulk_insert_is_idempotent_within_and_across_batches(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    now = datetime.now(UTC)
    rows = [
        Interaction(
            event_id=f"e{i % 700}",
            user_id="u",
            item_id=f"i{i % 7}",
            type=InteractionType.VIEW,
            occurred_at=now,
        )
        for i in range(1400)
    ]
    assert c.repository.add_interactions("t", rows) == 700
    assert c.repository.add_interactions("t", rows) == 0
    c.stop()


def test_keyset_pagination_returns_every_row_once_even_with_equal_timestamps(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    same = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        Interaction(
            event_id=f"e{i:04d}",
            user_id=f"u{i % 3}",
            item_id="x",
            type=InteractionType.VIEW,
            occurred_at=same + timedelta(seconds=i // 50),
        )
        for i in range(523)
    ]
    c.repository.add_interactions("t", rows)
    seen = list(c.repository.iter_interactions("t", batch_size=40))
    assert len(seen) == 523
    assert [r[4] for r in seen] == sorted(r[4] for r in seen)
    since = list(c.repository.iter_interactions("t", since=same + timedelta(seconds=5), batch_size=40))
    assert len(since) == 523 - 250
    c.stop()


def test_concurrent_upserts_of_the_same_item_do_not_fail(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            c.ingestion.ingest("t", [Item(id="same", title="Shared", body=f"version {n} of the text")])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert c.repository.get_item("t", "same") is not None
    c.stop()


def test_user_erasure_removes_interactions(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    c.repository.add_interactions(
        "t",
        [
            Interaction(event_id=f"e{i}", user_id="gone", item_id="x", type=InteractionType.VIEW)
            for i in range(5)
        ],
    )
    assert c.repository.delete_user_interactions("t", "gone") == 5
    assert c.repository.user_interactions("t", "gone") == []
    c.stop()


def test_fold_in_reuses_cached_gram_matrix() -> None:
    import numpy as np
    from scipy import sparse

    m = sparse.csr_matrix(np.eye(6, dtype=np.float32))
    als = ImplicitALS(ALSConfig(factors=3, iterations=2)).fit(m)
    first = als.item_gram()
    assert als.item_gram() is first
    als.fold_in(np.array([0]), np.array([1.0]))
    assert als.item_gram() is first
    with pytest.raises(RuntimeError):
        ImplicitALS(ALSConfig()).item_gram()


def test_api_rejects_malformed_events_with_422(api: TestClient) -> None:
    body = json.dumps({"events": [{"event_id": "x", "type": "item.upserted", "data": {"nope": 1}}]}).encode()
    r = api.post("/v1/events", content=body, headers={"Signature": sign_payload(SECRET, body)})
    assert r.status_code == 422 and r.json()["code"] == "validation_failed"
    garbage = b"not json"
    r = api.post("/v1/events", content=garbage, headers={"Signature": sign_payload(SECRET, garbage)})
    assert r.status_code == 422


def test_request_id_is_sanitised(api: TestClient) -> None:
    good = api.get("/v1/health", headers={"x-request-id": "abc.123"})
    assert good.headers["x-request-id"] == "abc.123"
    bad = api.get("/v1/health", headers={"x-request-id": "evil\nInjected: yes"})
    assert bad.headers["x-request-id"] != "evil\nInjected: yes" and len(bad.headers["x-request-id"]) == 32


def test_erasure_endpoint(api: TestClient) -> None:
    api.post("/v1/feedback", json={"principal": {"user_id": "leaver"}, "target_id": "x", "signal": "click"})
    r = api.delete("/v1/users/leaver/interactions")
    assert r.status_code == 200 and r.json()["interactions_deleted"] == 1


# Security primitives ---------------------------------------------------------------------
def test_rate_limiter_blocks_then_refills(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr("vantix.security.auth.time.monotonic", lambda: clock["t"])
    limiter = RateLimiter(per_minute=2)
    limiter.check("k")
    limiter.check("k")
    from vantix.errors import RateLimited

    with pytest.raises(RateLimited):
        limiter.check("k")
    clock["t"] += 30
    limiter.check("k")
    RateLimiter(0).check("k")


def test_signature_expiry_and_admin_requirement() -> None:
    from vantix.errors import AuthenticationError, PermissionDenied

    old = sign_payload("s", b"{}", timestamp=int(time.time()) - 3600)
    with pytest.raises(AuthenticationError, match="tolerance"):
        verify_signature("s", b"{}", old)
    with pytest.raises(AuthenticationError, match="Malformed"):
        verify_signature("s", b"{}", "garbage")
    auth = ApiKeyAuthenticator({"k": "t"}, {"a": "ops"})
    with pytest.raises(PermissionDenied):
        auth.require_admin(auth.authenticate("Bearer k"))
    assert auth.authenticate("Bearer a").is_admin


def test_settings_reject_malformed_key_spec() -> None:
    from vantix.config import Settings

    with pytest.raises(ValueError):
        Settings(api_keys="missing_separator").tenant_keys()
    assert Settings.parse_weights("a:0.5, b:1") == {"a": 0.5, "b": 1.0}
    assert Settings(cors_origins="https://a, https://b").cors_origins == ["https://a", "https://b"]  # type: ignore[arg-type]


# Cache, logging, scheduler ------------------------------------------------------------
def test_ttl_cache_expiry_and_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr("vantix.cache.time.monotonic", lambda: clock["t"])
    cache: TTLCache[int] = TTLCache("test", ttl_seconds=10, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None and cache.get("c") == 3
    clock["t"] = 11
    assert cache.get("c") is None
    cache.set("d", 4)
    cache.clear()
    assert cache.get("d") is None
    disabled: TTLCache[int] = TTLCache("off", ttl_seconds=0)
    disabled.set("x", 1)
    assert disabled.get("x") is None


def test_json_log_formatter_includes_context() -> None:
    token_r = request_id_var.set("rid1")
    token_t = tenant_id_var.set("acme")
    record = logging.LogRecord("vantix", logging.INFO, __file__, 1, "hello", (), None)
    record.custom = "value"
    payload = json.loads(JsonFormatter().format(record))
    request_id_var.reset(token_r)
    tenant_id_var.reset(token_t)
    assert payload["message"] == "hello" and payload["request_id"] == "rid1"
    assert payload["tenant_id"] == "acme" and payload["custom"] == "value"


def test_scheduler_trains_dirty_tenants_in_background(tmp_path: Path) -> None:
    c = Container.build(make_settings(tmp_path))
    c.ingestion.ingest("t", [Item(id="a", title="A", body="alpha"), Item(id="b", title="B", body="beta")])
    c.scheduler.interval = 0.05  # type: ignore[assignment]
    c.scheduler.mark_dirty("t")
    c.scheduler.start()
    wait_for(lambda: c.registry.get("t").trained_at.year > 1970, message="background training")
    c.scheduler.stop()
    assert c.registry.get("t").trained_at.year > 1970
    c.stop()


# CLI and client ---------------------------------------------------------------------------
def test_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vantix import cli
    from vantix.config import get_settings

    settings = make_settings(tmp_path)
    monkeypatch.setenv("VANTIX_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("VANTIX_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VANTIX_LOG_LEVEL", "WARNING")
    for key, value in settings.model_dump().items():
        if key.startswith(("vector_backend", "qdrant_", "lexical_backend", "opensearch_")):
            monkeypatch.setenv(f"VANTIX_{key.upper()}", str(getattr(value, "value", value)))
    get_settings.cache_clear()
    try:
        assert cli.main(["ingest", str(DATA / "items.jsonl"), "cli"]) == 0
        assert cli.main(["events", str(DATA / "events.jsonl"), "cli"]) == 0
        assert cli.main(["train", "cli"]) == 0
        assert cli.main(["reindex", "cli"]) == 0
        assert cli.main(["stats", "cli"]) == 0
        assert cli.main(["evaluate", "retrieval", str(DATA / "eval_queries.jsonl"), "cli"]) == 0
        out = capsys.readouterr().out
        assert '"indexed": 48' in out and '"interactions_recorded": 1218' in out and "ndcg@10" in out
        assert cli.main(["reindex", "cli", "full"]) == 0
        assert cli.main([]) == 2
        assert cli.main(["ingest"]) == 2
        assert cli.main(["evaluate"]) == 2
    finally:
        get_settings.cache_clear()


def test_python_client_against_app(tmp_path: Path) -> None:
    from vantix.api.app import create_app

    app = create_app(make_settings(tmp_path, webhook_secret=SECRET))
    with TestClient(app) as http:
        client = VantixClient("http://testserver", api_key=API_KEY, webhook_secret=SECRET)
        client._client = http
        http.headers.update({"Authorization": f"Bearer {API_KEY}"})
        items = [json.loads(line) for line in (DATA / "items.jsonl").read_text().splitlines()]
        assert client.upsert_items(items[:10], wait=True)["results"]
        job = client.upsert_items(items[10:12])
        wait_for(lambda: client.job(job["job_id"])["status"] == "completed", message="client ingestion job")
        assert client.track("amara", items[0]["id"], "view")["interactions_recorded"] == 1
        assert client.search(PRINCIPAL, "rollback")["hits"] is not None
        assert client.ask(PRINCIPAL, "How long does a rollback take?")["answer"]
        events = list(client.ask_stream(PRINCIPAL, "How long does a rollback take?"))
        assert events[0][0] == "sources" and events[-1][0] == "done"
        assert client.recommend(PRINCIPAL, "documents")["kind"] == "documents"
        assert "documents" in client.assist(PRINCIPAL, item_id=items[0]["id"])
        assert client.feedback(PRINCIPAL, items[0]["id"], "upvote")["recorded"]
        assert client.delete_item(items[0]["id"])["status"] == "deleted"
        with pytest.raises(VantixClientError) as info:
            client.job("missing")
        assert info.value.status == 404


def test_concurrent_schema_initialisation_is_safe(tmp_path: Path) -> None:
    from vantix.storage import Repository

    url = make_settings(tmp_path).database_url
    errors: list[Exception] = []

    def init() -> None:
        try:
            repo = Repository(url)
            repo.init_schema()
            assert repo.ping()
            repo.engine.dispose()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=init) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_cli_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vantix import cli
    from vantix.config import get_settings

    monkeypatch.setenv("VANTIX_DATABASE_URL", make_settings(tmp_path).database_url)
    get_settings.cache_clear()
    try:
        assert cli.main(["migrate"]) == 0
    finally:
        get_settings.cache_clear()


def test_worker_mode_detects_changes_from_other_processes_and_api_hot_reloads(tmp_path: Path) -> None:
    from vantix.recommender.scheduler import TrainingScheduler

    settings = make_settings(tmp_path, model_refresh_seconds=0)
    api_side = Container.build(settings)  # stands in for an API replica
    worker_side = Container.build(settings)  # stands in for the dedicated worker
    worker = TrainingScheduler(
        worker_side.trainer, worker_side.registry, worker_side.repository, 60, poll_database=True
    )
    api_side.ingestion.ingest(
        "t", [Item(id="a", title="A", body="alpha"), Item(id="b", title="B", body="beta")]
    )
    api_side.repository.add_interactions(
        "t", [Interaction(event_id="e1", user_id="u", item_id="a", type=InteractionType.VIEW)]
    )
    assert worker.tenants_due() == ["t"]
    assert worker.run_cycle() == ["t"]
    assert worker.tenants_due() == [], "nothing new since the last training run"
    assert api_side.registry.get("t").interaction_count == 1, "API replica hot reloads the new model"

    api_side.repository.add_interactions(
        "t", [Interaction(event_id="e2", user_id="u", item_id="b", type=InteractionType.VIEW)]
    )
    assert worker.tenants_due() == ["t"]
    worker.run_cycle()
    assert api_side.registry.get("t").interaction_count == 2
    api_side.stop()
    worker_side.stop()


def test_cli_health_reads_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vantix import cli
    from vantix.config import get_settings

    beat = tmp_path / "beat"
    monkeypatch.setenv("VANTIX_WORKER_HEARTBEAT_FILE", str(beat))
    get_settings.cache_clear()
    try:
        assert cli.main(["health"]) == 1
        beat.write_text("1")
        assert cli.main(["health"]) == 0
    finally:
        get_settings.cache_clear()
