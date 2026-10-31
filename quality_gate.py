#!/usr/bin/env python3
"""Model quality gate.

Loads the sample workspace into an isolated store, runs the retrieval and recommender
evaluations and compares every metric with scripts/quality_thresholds.json. Fails the build
when a change degrades ranking quality below the agreed floor. Writes a Markdown report
(for the GitHub job summary) and a JSON report (as a build artifact).

Usage:
    python scripts/quality_gate.py [REPORT_DIR]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = json.loads((ROOT / "scripts" / "quality_thresholds.json").read_text())
DATA = ROOT / "examples" / "data"
TENANT = "quality_gate"


def main() -> int:
    from vantix.config import Settings
    from vantix.container import Container
    from vantix.domain.models import Item
    from vantix.evaluation.harness import evaluate_recommender, evaluate_retrieval
    from vantix.events.schemas import EventBatch

    report_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            database_url=f"sqlite:///{tmp}/gate.db",
            model_dir=Path(tmp) / "models",
            rec_training_interval_seconds=0,
            retrieval_cache_ttl_seconds=0,
            log_level="WARNING",
            log_json=False,
        )
        container = Container.build(settings)
        container.start()
        items = [
            Item.model_validate(json.loads(line)) for line in (DATA / "items.jsonl").read_text().splitlines()
        ]
        container.ingestion.ingest(TENANT, items)
        events = [json.loads(line) for line in (DATA / "events.jsonl").read_text().splitlines()]
        for start in range(0, len(events), 1000):
            container.events.process(
                TENANT, EventBatch.model_validate({"events": events[start : start + 1000]})
            )
        container.queue.drain()
        container.scheduler.train_now(TENANT)
        results = {
            "retrieval": evaluate_retrieval(container, TENANT, DATA / "eval_queries.jsonl")["metrics"],
            "recommender": evaluate_recommender(container, TENANT)["metrics"],
        }
        container.stop()

    rows: list[tuple[str, str, float, float, bool]] = []
    for suite, floors in THRESHOLDS.items():
        if suite not in results:
            continue
        for metric, floor in floors.items():
            value = float(results[suite].get(metric, 0.0))
            rows.append((suite, metric, value, float(floor), value >= floor))

    failed = [r for r in rows if not r[4]]
    lines = [
        "## Model quality gate",
        "",
        "| Suite | Metric | Value | Floor | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {s} | {m} | {v:.4f} | {f:.4f} | {'pass' if ok else 'FAIL'} |" for s, m, v, f, ok in rows]
    lines += [
        "",
        f"**{'FAILED' if failed else 'PASSED'}**: {len(rows) - len(failed)} of {len(rows)} checks met.",
    ]
    markdown = "\n".join(lines)
    (report_dir / "quality_gate.md").write_text(markdown + "\n")
    (report_dir / "quality_gate.json").write_text(
        json.dumps({"results": results, "thresholds": THRESHOLDS}, indent=2)
    )
    print(markdown)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
