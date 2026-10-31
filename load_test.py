#!/usr/bin/env python3
"""Concurrent load test with latency and error budgets.

Sends a realistic mix of search, ask, recommendation and assist requests from many workers
and fails when p95 latency or the error rate exceeds the budget. Used by the nightly
pipeline against a freshly built container, and safe to point at staging.

Usage:
    python scripts/load_test.py BASE_URL API_KEY [REQUESTS] [CONCURRENCY]

Budgets come from scripts/quality_thresholds.json ("load" section).
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

THRESHOLDS = Path(__file__).with_name("quality_thresholds.json")
QUERIES = [
    "how do I roll back a deployment",
    "refresh token reuse",
    "API rate limit",
    "mobile release checklist",
    "data retention policy",
    "who approves warehouse access",
]
PRINCIPALS = [
    {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng"]},
    {"user_id": "chloe", "project_ids": ["p_mobile"], "group_ids": ["eng"]},
    {"user_id": "eve", "project_ids": ["p_data"], "group_ids": ["data"]},
]


def request_mix(rng: random.Random) -> tuple[str, str, dict]:
    principal = rng.choice(PRINCIPALS)
    roll = rng.random()
    if roll < 0.45:
        return "search", "/v1/search", {"principal": principal, "query": rng.choice(QUERIES)}
    if roll < 0.65:
        return (
            "recommend",
            "/v1/recommendations",
            {"principal": principal, "kind": rng.choice(["documents", "tasks"])},
        )
    if roll < 0.85:
        return "ask", "/v1/ask", {"principal": principal, "question": rng.choice(QUERIES)}
    return "assist", "/v1/assist", {"principal": principal, "item_id": "t_canary_alerts"}


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    base_url, api_key = args[0], args[1]
    total = int(args[2]) if len(args) > 2 else 400
    concurrency = int(args[3]) if len(args) > 3 else 16
    budget = json.loads(THRESHOLDS.read_text())["load"]

    client = httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
        limits=httpx.Limits(max_connections=concurrency * 2),
    )
    rng = random.Random(42)
    plan = [request_mix(rng) for _ in range(total)]

    def call(entry: tuple[str, str, dict]) -> tuple[str, float, bool]:
        name, path, payload = entry
        start = time.perf_counter()
        try:
            ok = client.post(path, json=payload).status_code == 200
        except httpx.HTTPError:
            ok = False
        return name, (time.perf_counter() - start) * 1000, ok

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(call, plan))
    elapsed = time.perf_counter() - started

    latencies = sorted(r[1] for r in results)
    errors = sum(1 for r in results if not r[2])
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    error_rate = errors / len(results)
    report = {
        "requests": len(results),
        "concurrency": concurrency,
        "throughput_rps": round(len(results) / elapsed, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "error_rate": round(error_rate, 4),
        "by_endpoint_p95_ms": {
            name: round(
                sorted(r[1] for r in results if r[0] == name)[
                    max(0, int(0.95 * sum(1 for r in results if r[0] == name)) - 1)
                ],
                1,
            )
            for name in {r[0] for r in results}
        },
        "budget": budget,
    }
    print(json.dumps(report, indent=2))
    failed = p95 > budget["p95_ms"] or error_rate > budget["max_error_rate"]
    if failed:
        print("LOAD TEST FAILED: budget exceeded", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
