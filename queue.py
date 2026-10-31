"""Bounded background ingestion queue with job tracking.

Webhooks and bulk loads return immediately with a job identifier while worker threads
chunk, embed and index in the background. Job progress is persisted, so a host platform
can poll ``GET /v1/jobs/{job_id}`` from any replica.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from vantix.domain.models import Item
from vantix.errors import QueueFull
from vantix.ingestion.pipeline import IngestionService
from vantix.observability.logging import get_logger
from vantix.observability.metrics import QUEUE_DEPTH
from vantix.storage.repository import Repository

log = get_logger(__name__)


@dataclass(slots=True)
class _Work:
    job_id: str
    tenant_id: str
    items: list[Item]
    deletes: list[str]


class IngestionQueue:
    def __init__(
        self,
        service: IngestionService,
        repository: Repository,
        workers: int = 2,
        max_size: int = 10_000,
        batch_size: int = 32,
    ) -> None:
        self.service = service
        self.repo = repository
        self.batch_size = batch_size
        self._queue: queue.Queue[_Work | None] = queue.Queue(maxsize=max_size)
        self._threads = [
            threading.Thread(target=self._run, name=f"vantix_ingest_{i}", daemon=True) for i in range(workers)
        ]
        self._started = False
        self._on_complete: list = []  # type: ignore[type-arg]

    def on_complete(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._on_complete.append(callback)

    def start(self) -> None:
        if self._started:
            return
        for thread in self._threads:
            thread.start()
        self._started = True

    def stop(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._started = False

    def submit(self, tenant_id: str, items: list[Item], deletes: list[str] | None = None) -> str:
        deletes = deletes or []
        job_id = self.repo.create_job(tenant_id, total=len(items) + len(deletes))
        try:
            self._queue.put_nowait(_Work(job_id, tenant_id, items, deletes))
        except queue.Full as exc:
            self.repo.update_job(job_id, status="rejected", error={"error": "queue_full"})
            raise QueueFull("Ingestion queue is full; retry with backoff") from exc
        QUEUE_DEPTH.set(self._queue.qsize())
        return job_id

    def drain(self, timeout: float = 30.0) -> None:
        """Block until all queued work is processed (used by tests and the CLI)."""
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        done.wait(timeout)

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    return
                self._process(work)
            finally:
                self._queue.task_done()
                QUEUE_DEPTH.set(self._queue.qsize())

    def _process(self, work: _Work) -> None:
        self.repo.update_job(work.job_id, status="running")
        failed_any = False
        try:
            for start in range(0, len(work.items), self.batch_size):
                batch = work.items[start : start + self.batch_size]
                for result in self.service.ingest(work.tenant_id, batch):
                    if result.status == "failed":
                        failed_any = True
                        self.repo.update_job(
                            work.job_id,
                            failed_delta=1,
                            error={"item_id": result.item_id, "error": result.error},
                        )
                    else:
                        self.repo.update_job(work.job_id, processed_delta=1)
            for item_id in work.deletes:
                self.service.delete(work.tenant_id, item_id)
                self.repo.update_job(work.job_id, processed_delta=1)
            self.repo.update_job(work.job_id, status="completed_with_errors" if failed_any else "completed")
        except Exception as exc:  # noqa: BLE001
            log.exception("ingestion_job_failed", extra={"job_id": work.job_id})
            self.repo.update_job(work.job_id, status="failed", error={"error": str(exc)})
        for callback in self._on_complete:
            try:
                callback(work.tenant_id)
            except Exception:  # noqa: BLE001
                log.exception("ingestion_callback_failed")
