"""Background retraining scheduler.

Retrains each tenant's model on a fixed interval, but only when new interactions or content
have arrived since the previous run. Training happens off the request path and the new
model is swapped in atomically.

Two modes:

* In process (default for a single node): changes seen by this process mark tenants dirty.
* Database polling (``poll_database=True``, used by ``vantix worker``): the worker compares
  each tenant's current content version and interaction count with the snapshot stored in
  its latest model, so it notices changes received by any API replica.
"""

from __future__ import annotations

import threading
import time

from vantix.observability.logging import get_logger
from vantix.recommender.model import ModelRegistry, RecommenderTrainer
from vantix.storage.repository import Repository

log = get_logger(__name__)


class TrainingScheduler:
    def __init__(
        self,
        trainer: RecommenderTrainer,
        registry: ModelRegistry,
        repository: Repository,
        interval_seconds: int,
        poll_database: bool = False,
    ) -> None:
        self.trainer = trainer
        self.registry = registry
        self.repo = repository
        self.interval = interval_seconds
        self.poll_database = poll_database
        self.last_cycle_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dirty: set[str] = set()
        self._lock = threading.Lock()
        self._train_locks: dict[str, threading.Lock] = {}

    def mark_dirty(self, tenant_id: str) -> None:
        with self._lock:
            self._dirty.add(tenant_id)

    def train_now(self, tenant_id: str) -> None:
        with self._lock:
            lock = self._train_locks.setdefault(tenant_id, threading.Lock())
        with lock:
            model = self.trainer.train(tenant_id)
            self.registry.put(model)
        with self._lock:
            self._dirty.discard(tenant_id)

    def start(self) -> None:
        if self.interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="vantix_trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def tenants_due(self) -> list[str]:
        with self._lock:
            due = set(self._dirty)
        if self.poll_database:
            for tenant_id in self.repo.list_tenants():
                model = self.registry.get(tenant_id)
                if model.content_version != self.repo.content_version(
                    tenant_id
                ) or model.interaction_count != self.repo.interaction_count(tenant_id):
                    due.add(tenant_id)
        return sorted(due)

    def run_cycle(self) -> list[str]:
        """Train every tenant that is due. Returns the tenants trained successfully."""
        trained: list[str] = []
        for tenant_id in self.tenants_due():
            try:
                self.train_now(tenant_id)
                trained.append(tenant_id)
            except Exception:  # noqa: BLE001
                log.exception("scheduled_training_failed", extra={"tenant": tenant_id})
        self.last_cycle_at = time.time()
        return trained

    def run_forever(self) -> None:
        """Foreground loop used by ``vantix worker``. Returns after ``stop`` is called."""
        self.run_cycle()
        while not self._stop.wait(self.interval):
            self.run_cycle()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.run_cycle()
