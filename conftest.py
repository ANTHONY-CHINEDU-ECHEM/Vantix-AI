from __future__ import annotations

import json
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


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "database_url": f"sqlite:///{tmp_path / 'vantix.db'}",
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
