from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from tests.conftest import ADMIN_KEY, SECRET, load_events, load_items
from vantix.security.auth import sign_payload, verify_signature

PRINCIPAL = {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng", "leads"]}


def seed(api: TestClient) -> None:
    items = [i.model_dump(mode="json") for i in load_items()]
    r = api.post("/v1/items", json={"items": items, "wait": True})
    assert r.status_code == 200, r.text
    assert {x["status"] for x in r.json()["results"]} == {"indexed"}
    events = load_events()
    body = json.dumps({"events": events[:1000]}).encode()
    r = api.post(
        "/v1/events",
        content=body,
        headers={"Signature": sign_payload(SECRET, body), "Content-Type": "application/json"},
    )
    assert r.status_code == 202, r.text
    r = api.post("/v1/admin/train")
    assert r.status_code == 200 and r.json()["collaborative_model"] is True


def test_health_ready_metrics(api: TestClient) -> None:
    assert api.get("/v1/health").json()["status"] == "ok"
    assert api.get("/v1/ready").json()["ready"] is True
    assert "vantix_http_requests_total" in api.get("/v1/metrics").text


def test_authentication_and_tenancy(api: TestClient) -> None:
    assert api.post("/v1/search", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert api.post("/v1/search", json={}, headers={"Authorization": ""}).status_code == 401
    seed(api)
    other = {"Authorization": "Bearer other_key"}
    r = api.post("/v1/search", json={"principal": PRINCIPAL, "query": "rollback"}, headers=other)
    assert r.status_code == 200 and r.json()["hits"] == [], "tenants must be isolated"
    assert api.get("/v1/admin/stats", params={"tenant_id": "other"}).status_code == 403
    admin = {"Authorization": f"Bearer {ADMIN_KEY}"}
    assert api.get("/v1/admin/stats", params={"tenant_id": "acme"}, headers=admin).json()["chunks"] > 0


def test_end_to_end_flow(api: TestClient) -> None:
    seed(api)
    r = api.post("/v1/search", json={"principal": PRINCIPAL, "query": "refresh token reused", "k": 3})
    assert r.status_code == 200
    assert r.json()["hits"][0]["item_id"] == "kb_oauth"

    r = api.post("/v1/ask", json={"principal": PRINCIPAL, "question": "What is the public API rate limit?"})
    body = r.json()
    assert (
        r.status_code == 200 and body["grounded"] and body["citations"][0]["item_id"] == "kb_api_rate_limits"
    )

    r = api.post("/v1/recommendations", json={"principal": PRINCIPAL, "kind": "tasks", "k": 5})
    assert r.status_code == 200 and r.json()["recommendations"]
    assert r.json()["model_trained_at"] is not None

    r = api.post(
        "/v1/recommendations",
        json={"principal": PRINCIPAL, "kind": "experts", "context_item_id": "t_index_events_table"},
    )
    assert r.status_code == 200 and r.json()["recommendations"]

    r = api.post(
        "/v1/assist",
        json={"principal": PRINCIPAL, "item_id": "t_canary_alerts", "question": "When do we halt a canary?"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["answer"]["citations"] and payload["documents"] and payload["experts"]

    r = api.post(
        "/v1/feedback", json={"principal": PRINCIPAL, "target_id": "kb_monitoring", "signal": "dismiss"}
    )
    assert r.json()["recorded"] is True
    r = api.post("/v1/recommendations", json={"principal": PRINCIPAL, "kind": "documents", "k": 50})
    assert "kb_monitoring" not in [x["target_id"] for x in r.json()["recommendations"]]


def test_streaming_answer(api: TestClient) -> None:
    seed(api)
    with api.stream(
        "POST",
        "/v1/ask",
        json={"principal": PRINCIPAL, "question": "How long does rollback take?", "stream": True},
    ) as r:
        text = "".join(r.iter_text())
    assert "event: sources" in text and "event: done" in text


def test_async_ingestion_job_and_events(api: TestClient) -> None:
    item = {
        "id": "new_doc",
        "title": "Brand new runbook",
        "body": "Restart the worker with care.",
        "project_id": "p_platform",
    }
    r = api.post("/v1/items", json={"items": [item]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = api.get(f"/v1/jobs/{job_id}").json()
        if job["status"].startswith("completed"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed" and job["processed"] == 1
    assert api.get("/v1/items/new_doc").json()["title"] == "Brand new runbook"

    events = {"events": [{"event_id": "del_1", "type": "item.deleted", "data": {"id": "new_doc"}}]}
    body = json.dumps(events).encode()
    headers = {"Signature": sign_payload(SECRET, body), "Content-Type": "application/json"}
    first = api.post("/v1/events", content=body, headers=headers).json()
    assert first["deletes_queued"] == 1
    again = api.post(
        "/v1/events", content=body, headers={**headers, "Signature": sign_payload(SECRET, body)}
    ).json()
    assert again["duplicates"] == 1, "events must be idempotent"


def test_webhook_signature_rejected(api: TestClient) -> None:
    body = json.dumps({"events": [{"event_id": "x", "type": "item.deleted", "data": {"id": "y"}}]}).encode()
    r = api.post(
        "/v1/events", content=body, headers={"Signature": "t=1,v1=bad", "Content-Type": "application/json"}
    )
    assert r.status_code == 401
    r = api.post("/v1/events", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 401


def test_signature_helpers() -> None:
    header = sign_payload("s", b"{}")
    verify_signature("s", b"{}", header)


def test_validation_errors(api: TestClient) -> None:
    r = api.post("/v1/recommendations", json={"principal": PRINCIPAL, "kind": "unknown"})
    assert r.status_code == 422
    r = api.post("/v1/items", json={"items": [{"id": "x", "unexpected": True}]})
    assert r.status_code == 422
    assert api.get("/v1/jobs/missing").status_code == 404
