#!/usr/bin/env python3
"""Post deployment smoke test for a running Vantix instance.

Verifies liveness, readiness, authentication, ingestion, search, grounded answering,
recommendations, feedback and deletion end to end, using a dedicated probe item that is
removed afterwards. Exits non zero on the first failure, so CD can roll back automatically.

Usage:
    python scripts/smoke_test.py BASE_URL API_KEY [WEBHOOK_SECRET]

Environment alternatives: VANTIX_SMOKE_URL, VANTIX_SMOKE_API_KEY, VANTIX_SMOKE_WEBHOOK_SECRET.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import uuid

import httpx

PROBE_PROJECT = "vantix_smoke"


class SmokeFailure(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ok  {message}")


def wait_until_ready(client: httpx.Client, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            response = client.get("/v1/ready")
            if response.status_code == 200 and response.json().get("ready"):
                return
            last = response.text
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(2)
    raise SmokeFailure(f"service not ready within {timeout}s: {last[:300]}")


def run(base_url: str, api_key: str, webhook_secret: str | None) -> None:
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=60.0)
    auth = {"Authorization": f"Bearer {api_key}"}
    marker = uuid.uuid4().hex[:10]
    item_id = f"smoke_{marker}"
    principal = {"user_id": f"smoke_user_{marker}", "project_ids": [PROBE_PROJECT]}

    print(f"Smoke testing {base_url}")
    health = client.get("/v1/health")
    check(health.status_code == 200 and health.json()["status"] == "ok", "liveness responds")
    wait_until_ready(client)
    check(True, "readiness reports every dependency healthy")
    check(
        client.post("/v1/search", json={}, headers={"Authorization": "Bearer invalid"}).status_code == 401,
        "invalid API keys are rejected",
    )

    item = {
        "id": item_id,
        "kind": "document",
        "title": f"Smoke probe {marker}",
        "body": f"The smoke probe {marker} confirms that the quokka deployment pipeline is healthy.",
        "project_id": PROBE_PROJECT,
        "acl": {"visibility": "project"},
    }
    upsert = client.post("/v1/items", json={"items": [item], "wait": True}, headers=auth)
    check(
        upsert.status_code == 200 and upsert.json()["results"][0]["status"] in ("indexed", "unchanged"),
        "synchronous ingestion indexes the probe item",
    )

    try:
        search = client.post(
            "/v1/search", json={"principal": principal, "query": f"quokka {marker}", "k": 3}, headers=auth
        )
        check(
            search.status_code == 200 and any(h["item_id"] == item_id for h in search.json()["hits"]),
            "search finds the probe item",
        )

        outsider = {"user_id": "smoke_outsider", "project_ids": []}
        hidden = client.post(
            "/v1/search", json={"principal": outsider, "query": f"quokka {marker}"}, headers=auth
        )
        check(
            all(h["item_id"] != item_id for h in hidden.json()["hits"]), "access control hides the probe item"
        )

        ask = client.post(
            "/v1/ask",
            json={"principal": principal, "question": f"What does probe {marker} confirm?"},
            headers=auth,
        )
        body = ask.json()
        check(
            ask.status_code == 200
            and body["grounded"]
            and any(c["item_id"] == item_id for c in body["citations"]),
            "answers are grounded and cite the probe item",
        )

        recs = client.post(
            "/v1/recommendations", json={"principal": principal, "kind": "documents", "k": 3}, headers=auth
        )
        check(recs.status_code == 200 and "recommendations" in recs.json(), "recommendations respond")

        event_body = json.dumps(
            {
                "events": [
                    {
                        "event_id": f"smoke_evt_{marker}",
                        "type": "interaction.recorded",
                        "data": {"user_id": principal["user_id"], "item_id": item_id, "type": "view"},
                    }
                ]
            }
        ).encode()
        headers = dict(auth)
        if webhook_secret:
            ts = int(time.time())
            digest = hmac.new(
                webhook_secret.encode(), f"{ts}.".encode() + event_body, hashlib.sha256
            ).hexdigest()
            headers["Signature"] = f"t={ts},v1={digest}"
        events = client.post("/v1/events", content=event_body, headers=headers)
        check(
            events.status_code == 202 and events.json()["interactions_recorded"] == 1,
            "signed events are accepted",
        )

        metrics = client.get("/v1/metrics")
        check(
            metrics.status_code == 200 and "vantix_http_requests_total" in metrics.text,
            "metrics are exported",
        )
    finally:
        client.delete(f"/v1/items/{item_id}", headers=auth)
        client.delete(f"/v1/users/{principal['user_id']}/interactions", headers=auth)
    check(client.get(f"/v1/items/{item_id}", headers=auth).status_code == 404, "probe item is cleaned up")
    print("Smoke test passed")


def main() -> int:
    args = sys.argv[1:]
    base_url = args[0] if args else os.environ.get("VANTIX_SMOKE_URL", "")
    api_key = args[1] if len(args) > 1 else os.environ.get("VANTIX_SMOKE_API_KEY", "")
    secret = args[2] if len(args) > 2 else os.environ.get("VANTIX_SMOKE_WEBHOOK_SECRET") or None
    if not base_url or not api_key:
        print(__doc__)
        return 2
    try:
        run(base_url, api_key, secret)
    except (SmokeFailure, httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
