"""Reference integration for a host project management platform (for example ProWin).

Shows the four integration touchpoints end to end against a running Vantix service:

1. Backfill: bulk load existing tasks and documents.
2. Change data: forward item and interaction events with a signed webhook (outbox pattern).
3. Read paths: search, ask, recommendations and the task panel assist call.
4. Feedback: close the loop so recommendations learn from what users act on.

Run a local server first (``vantix serve``) and then: ``python examples/host_integration.py``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from vantix.client import VantixClient

BASE_URL = os.environ.get("VANTIX_URL", "http://localhost:8080")
API_KEY = os.environ.get("VANTIX_KEY", "demo_key")
WEBHOOK_SECRET = os.environ.get("VANTIX_WEBHOOK_SECRET") or None
DATA = Path(__file__).parent / "data"


def principal_for(user: dict) -> dict:
    """Map the host platform's user and membership model onto a Vantix principal."""
    return {
        "user_id": user["id"],
        "project_ids": user["project_ids"],
        "group_ids": user["team_ids"],
        "is_admin": user.get("role") == "workspace_admin",
    }


def main() -> None:
    client = VantixClient(BASE_URL, API_KEY, webhook_secret=WEBHOOK_SECRET)

    # 1. Backfill in batches; poll the job until it completes.
    items = [json.loads(line) for line in (DATA / "items.jsonl").read_text().splitlines()]
    for start in range(0, len(items), 500):
        job = client.upsert_items(items[start : start + 500])
        while client.job(job["job_id"])["status"] in ("queued", "running"):
            time.sleep(0.5)
    events = [json.loads(line) for line in (DATA / "events.jsonl").read_text().splitlines()]
    for start in range(0, len(events), 1000):
        client.send_events(events[start : start + 1000])

    # 2. Live change: a task was edited in the host UI. The host writes this event to its
    #    outbox table in the same transaction as the edit; a relay posts it to Vantix.
    client.send_events(
        [
            {
                "event_id": f"task_updated_{uuid.uuid4().hex}",
                "type": "item.upserted",
                "data": {
                    "id": "t_canary_alerts",
                    "kind": "task",
                    "title": "Add automatic canary abort on error spike",
                    "body": "Abort the canary when the error rate exceeds one percent for five minutes.",
                    "project_id": "p_platform",
                    "status": "in_review",
                    "assignee_ids": ["amara"],
                    "linked_item_ids": ["kb_deploy_runbook", "kb_monitoring"],
                },
            }
        ]
    )

    user = {"id": "amara", "project_ids": ["p_platform"], "team_ids": ["eng", "leads"]}
    principal = principal_for(user)

    # 3. Read paths.
    print(json.dumps(client.search(principal, "rollback a release", k=3), indent=2)[:800])
    print(json.dumps(client.ask(principal, "How long does a rollback take?"), indent=2)[:800])
    print(json.dumps(client.recommend(principal, "tasks", k=5), indent=2)[:800])
    panel = client.assist(principal, item_id="t_canary_alerts", question="When do we halt a canary?")
    print(json.dumps(panel, indent=2)[:1200])

    # 4. Feedback from the UI.
    client.feedback(principal, "kb_deploy_runbook", "click", surface="assist")
    client.close()


if __name__ == "__main__":
    main()
