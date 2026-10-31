"""Command line interface.

Commands use positional arguments only:

    vantix migrate
    vantix serve
    vantix worker
    vantix health
    vantix ingest PATH [TENANT]
    vantix events PATH [TENANT]
    vantix train [TENANT]
    vantix reindex [TENANT] [full]
    vantix stats [TENANT]
    vantix evaluate retrieval PATH [TENANT]
    vantix evaluate recommender [TENANT]
    vantix demo [TENANT]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from vantix.config import get_settings

DEFAULT_TENANT = "demo"


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _container():  # type: ignore[no-untyped-def]
    from vantix.container import Container
    from vantix.observability.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)
    container = Container.build(settings)
    return container


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_migrate(_args: list[str]) -> int:
    """Create or upgrade the database schema. Safe to run concurrently and repeatedly."""
    from vantix.storage import Repository

    repo = Repository(get_settings().database_url)
    repo.init_schema()
    repo.engine.dispose()
    _print({"migrated": True})
    return 0


def cmd_worker(_args: list[str]) -> int:
    """Run the dedicated training worker until SIGTERM or SIGINT.

    The worker polls the database for tenants whose content or activity changed since their
    last model, retrains them and persists models to the shared model directory, from which
    API replicas hot reload. A heartbeat file is touched after every cycle for liveness probes.
    """
    import signal
    import threading

    from vantix.recommender.scheduler import TrainingScheduler

    settings = get_settings()
    container = _container()
    interval = settings.rec_training_interval_seconds or 300
    scheduler = TrainingScheduler(
        container.trainer, container.registry, container.repository, interval, poll_database=True
    )
    heartbeat = Path(settings.worker_heartbeat_file)
    stop = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stop.set()
        scheduler.stop()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"worker": "started", "interval_seconds": interval}), flush=True)
    while not stop.is_set():
        trained = scheduler.run_cycle()
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.write_text(str(time.time()))
        if trained:
            print(json.dumps({"trained": trained}), flush=True)
        stop.wait(interval)
    container.stop()
    return 0


def cmd_health(_args: list[str]) -> int:
    """Exit 0 when the worker heartbeat is fresh. Used as a Kubernetes liveness command."""
    settings = get_settings()
    interval = settings.rec_training_interval_seconds or 300
    try:
        age = time.time() - Path(settings.worker_heartbeat_file).stat().st_mtime
    except OSError:
        print("no heartbeat")
        return 1
    healthy = age < max(3 * interval, 600)
    print(json.dumps({"heartbeat_age_seconds": round(age, 1), "healthy": healthy}))
    return 0 if healthy else 1


def cmd_serve(_args: list[str]) -> int:
    import uvicorn

    settings = get_settings()
    workers = settings.workers
    memory = settings.vector_backend.value == "memory" or settings.lexical_backend.value == "memory"
    if memory and workers > 1:
        # In memory indexes live inside each process, so several worker processes would each
        # see a different index. Use Qdrant and OpenSearch to scale beyond one process.
        print(
            json.dumps({"warning": "memory indexes require a single worker process", "workers": 1}),
            flush=True,
        )
        workers = 1
    uvicorn.run(
        "vantix.api.app:app_factory",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=workers,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
    )
    return 0


def cmd_ingest(args: list[str]) -> int:
    from vantix.domain.models import Item

    path, tenant = Path(args[0]), (args[1] if len(args) > 1 else DEFAULT_TENANT)
    container = _container()
    items = [Item.model_validate(r) for r in _load_records(path)]
    summary: dict[str, int] = {}
    for start in range(0, len(items), 64):
        for result in container.ingestion.ingest(tenant, items[start : start + 64]):
            summary[result.status] = summary.get(result.status, 0) + 1
    _print({"tenant_id": tenant, "items": len(items), "results": summary})
    container.stop()
    return 0


def cmd_events(args: list[str]) -> int:
    from vantix.events.schemas import EventBatch

    path, tenant = Path(args[0]), (args[1] if len(args) > 1 else DEFAULT_TENANT)
    container = _container()
    container.queue.start()
    records = _load_records(path)
    totals: dict[str, int] = {"accepted": 0, "duplicates": 0, "interactions_recorded": 0}
    for start in range(0, len(records), 1000):
        result = container.events.process(
            tenant, EventBatch.model_validate({"events": records[start : start + 1000]})
        )
        for key in totals:
            totals[key] += getattr(result, key)
    container.queue.drain(timeout=600)
    _print({"tenant_id": tenant, **totals})
    container.stop()
    return 0


def cmd_train(args: list[str]) -> int:
    tenant = args[0] if args else DEFAULT_TENANT
    container = _container()
    model = container.trainer.train(tenant)
    container.registry.put(model)
    _print(
        {
            "tenant_id": tenant,
            "trained_at": model.trained_at,
            "users": len(model.user_ids),
            "items": len(model.item_ids),
            "graph_nodes": model.graph.size if model.graph else 0,
        }
    )
    container.stop()
    return 0


def cmd_reindex(args: list[str]) -> int:
    tenant = args[0] if args else DEFAULT_TENANT
    full = len(args) > 1 and args[1] == "full"
    container = _container()
    if full:
        _print({"tenant_id": tenant, "items_reembedded": container.ingestion.reembed_all(tenant)})
    else:
        _print({"tenant_id": tenant, "chunks_indexed": container.ingestion.rebuild_indexes(tenant)})
    container.stop()
    return 0


def cmd_stats(args: list[str]) -> int:
    tenant = args[0] if args else DEFAULT_TENANT
    container = _container()
    _print({"tenant_id": tenant, **container.repository.stats(tenant)})
    container.stop()
    return 0


def cmd_evaluate(args: list[str]) -> int:
    from vantix.evaluation.harness import evaluate_recommender, evaluate_retrieval

    if not args or args[0] not in ("retrieval", "recommender"):
        print("usage: vantix evaluate retrieval PATH [TENANT] | vantix evaluate recommender [TENANT]")
        return 2
    container = _container()
    container.start()
    if args[0] == "retrieval":
        tenant = args[2] if len(args) > 2 else DEFAULT_TENANT
        _print(evaluate_retrieval(container, tenant, Path(args[1])))
    else:
        tenant = args[1] if len(args) > 1 else DEFAULT_TENANT
        _print(evaluate_recommender(container, tenant))
    container.stop()
    return 0


def cmd_demo(args: list[str]) -> int:
    tenant = args[0] if args else DEFAULT_TENANT
    root = Path(__file__).resolve().parents[2] / "examples" / "data"
    if not root.exists():
        root = Path.cwd() / "examples" / "data"
    cmd_ingest([str(root / "items.jsonl"), tenant])
    cmd_events([str(root / "events.jsonl"), tenant])
    cmd_train([tenant])
    return 0


COMMANDS = {
    "migrate": cmd_migrate,
    "serve": cmd_serve,
    "worker": cmd_worker,
    "health": cmd_health,
    "ingest": cmd_ingest,
    "events": cmd_events,
    "train": cmd_train,
    "reindex": cmd_reindex,
    "stats": cmd_stats,
    "evaluate": cmd_evaluate,
    "demo": cmd_demo,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    command = argv.pop(0)
    if command in ("ingest", "events") and not argv:
        print(f"usage: vantix {command} PATH [TENANT]")
        return 2
    return COMMANDS[command](argv)


if __name__ == "__main__":
    raise SystemExit(main())
