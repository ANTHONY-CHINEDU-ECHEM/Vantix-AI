"""Typed Python client for host platforms integrating with Vantix.

Example::

    from vantix.client import VantixClient

    client = VantixClient("https://vantix.internal", api_key="...", webhook_secret="...")
    principal = {"user_id": "u_42", "project_ids": ["p_web"], "group_ids": ["eng"]}
    recs = client.recommend(principal, kind="tasks", k=5)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from vantix.security.auth import sign_payload


class VantixClientError(RuntimeError):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"Vantix request failed with status {status}: {body}")
        self.status = status
        self.body = body


class VantixClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        webhook_secret: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self.webhook_secret = webhook_secret

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VantixClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._client.post(path, json=payload)
        return self._handle(response)

    @staticmethod
    def _handle(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            raise VantixClientError(response.status_code, body)
        return response.json()

    # Ingestion --------------------------------------------------------------------------
    def upsert_items(self, items: list[dict[str, Any]], wait: bool = False) -> Any:
        return self._post("/v1/items", {"items": items, "wait": wait})

    def delete_item(self, item_id: str) -> Any:
        return self._handle(self._client.delete(f"/v1/items/{item_id}"))

    def job(self, job_id: str) -> Any:
        return self._handle(self._client.get(f"/v1/jobs/{job_id}"))

    def send_events(self, events: list[dict[str, Any]]) -> Any:
        body = json.dumps({"events": events}, default=str).encode()
        headers = {"Content-Type": "application/json"}
        if self.webhook_secret:
            headers["Signature"] = sign_payload(self.webhook_secret, body)
        return self._handle(self._client.post("/v1/events", content=body, headers=headers))

    def track(self, user_id: str, item_id: str, interaction: str, **fields: Any) -> Any:
        event = {
            "event_id": fields.pop("event_id", uuid.uuid4().hex),
            "type": "interaction.recorded",
            "occurred_at": fields.pop("occurred_at", datetime.now(UTC).isoformat()),
            "data": {"user_id": user_id, "item_id": item_id, "type": interaction, **fields},
        }
        return self.send_events([event])

    # Retrieval and recommendation --------------------------------------------------------
    def search(self, principal: dict[str, Any], query: str, **options: Any) -> Any:
        return self._post("/v1/search", {"principal": principal, "query": query, **options})

    def ask(self, principal: dict[str, Any], question: str, **options: Any) -> Any:
        return self._post("/v1/ask", {"principal": principal, "question": question, **options})

    def ask_stream(
        self, principal: dict[str, Any], question: str, **options: Any
    ) -> Iterator[tuple[str, Any]]:
        payload = {"principal": principal, "question": question, "stream": True, **options}
        with self._client.stream("POST", "/v1/ask", json=payload) as response:
            if response.status_code >= 400:
                response.read()
                raise VantixClientError(response.status_code, response.text)
            event = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event:
                    yield event, json.loads(line[5:].strip())

    def recommend(self, principal: dict[str, Any], kind: str, k: int = 10, **options: Any) -> Any:
        return self._post("/v1/recommendations", {"principal": principal, "kind": kind, "k": k, **options})

    def assist(
        self,
        principal: dict[str, Any],
        item_id: str | None = None,
        question: str | None = None,
        **options: Any,
    ) -> Any:
        return self._post(
            "/v1/assist", {"principal": principal, "item_id": item_id, "question": question, **options}
        )

    def feedback(
        self,
        principal: dict[str, Any],
        target_id: str,
        signal: str,
        surface: str = "recommendation",
        **options: Any,
    ) -> Any:
        return self._post(
            "/v1/feedback",
            {"principal": principal, "target_id": target_id, "signal": signal, "surface": surface, **options},
        )
