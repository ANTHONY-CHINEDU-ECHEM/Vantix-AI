from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vantix.api.app import create_app
from vantix.config import Settings
from vantix.container import Container
from vantix.domain.models import Item
from vantix.events.schemas import EventBatch

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
TENANT = "acme"
API_KEY = "test_key"
ADMIN_KEY = "admin_key"
SECRET = "whsec_test"


# Optional real backends. When these are set (as they are in the CI integration job), the
# entire suite runs against PostgreSQL, Qdrant and OpenSearch instead of the local defaults.
PG_URL = os.environ.get("VANTIX_TEST_DATABASE_URL")
QDRANT_URL = os.environ.get("VANTIX_TEST_QDRANT_URL")
OPENSEARCH_URL = os.environ.get("VANTIX_TEST_OPENSEARCH_URL")


def isolated_database_url(tmp_path: Path) -> str:
    """SQLite file per test, or a fresh PostgreSQL schema per test when PostgreSQL is configured."""
    if not PG_URL:
        return f"sqlite:///{tmp_path / 'vantix.db'}"
    import sqlalchemy

    schema = f"t_{uuid.uuid4().hex[:12]}"
    engine = sqlalchemy.create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text(f'CREATE SCHEMA "{schema}"'))
    engine.dispose()
    separator = "&" if "?" in PG_URL else "?"
    return f"{PG_URL}{separator}options=-csearch_path%3D{schema}"


def backend_overrides(tmp_path: Path) -> dict[str, object]:
    suffix = uuid.uuid4().hex[:10]
    overrides: dict[str, object] = {}
    if QDRANT_URL:
        overrides.update(vector_backend="qdrant", qdrant_url=QDRANT_URL, qdrant_collection=f"vx_{suffix}")
    if OPENSEARCH_URL:
        overrides.update(
            lexical_backend="opensearch", opensearch_url=OPENSEARCH_URL, opensearch_index=f"vx_{suffix}"
        )
    return overrides


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "database_url": isolated_database_url(tmp_path),
        **backend_overrides(tmp_path),
        "model_dir": tmp_path / "models",
        "api_keys": f"{TENANT}:{API_KEY},other:other_key",
        "admin_api_keys": f"ops:{ADMIN_KEY}",
        "log_json": False,
        "log_level": "WARNING",
        "rec_training_interval_seconds": 0,
        "retrieval_cache_ttl_seconds": 0,
        "rate_limit_per_minute": 100000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05, message: str = "condition"):  # type: ignore[no-untyped-def]
    """Poll until ``predicate`` returns a truthy value. Generous timeouts keep slow CI runners green."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {message}")
        time.sleep(interval)


def load_items() -> list[Item]:
    return [Item.model_validate(json.loads(line)) for line in (DATA / "items.jsonl").read_text().splitlines()]


def load_events() -> list[dict]:
    return [json.loads(line) for line in (DATA / "events.jsonl").read_text().splitlines()]


@pytest.fixture(scope="session")
def seeded_container(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Container]:
    tmp = tmp_path_factory.mktemp("seeded")
    container = Container.build(make_settings(tmp))
    container.start()
    container.ingestion.ingest(TENANT, load_items())
    events = load_events()
    for start in range(0, len(events), 1000):
        container.events.process(TENANT, EventBatch.model_validate({"events": events[start : start + 1000]}))
    container.scheduler.train_now(TENANT)
    yield container
    container.stop()


@pytest.fixture()
def api(tmp_path: Path) -> Iterator[TestClient]:
    settings = make_settings(tmp_path, webhook_secret=SECRET)
    app = create_app(settings)
    with TestClient(app) as client:
        client.headers.update({"Authorization": f"Bearer {API_KEY}"})
        yield client
