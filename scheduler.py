"""Background retraining scheduler.

Retrains each tenant's model on a fixed interval, but only when new interactions or content
have arrived since the previous run. Training happens off the request path and the new
model is swapped in atomically.
"""

from __future__ import annotations

import threading

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
    ) -> None:
        self.trainer = trainer
        self.registry = registry
        self.repo = repository
        self.interval = interval_seconds
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

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                tenants = list(self._dirty)
            for tenant_id in tenants:
                try:
                    self.train_now(tenant_id)
                except Exception:  # noqa: BLE001
                    log.exception("scheduled_training_failed", extra={"tenant": tenant_id})
