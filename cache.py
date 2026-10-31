"""Thread safe TTL and LRU cache with per tenant versioned keys.

Keys embed the tenant content version, so any ingestion instantly invalidates cached search
results for that tenant without explicit purges and without cross tenant interference.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

from vantix.observability.metrics import CACHE_HITS, CACHE_MISSES

V = TypeVar("V")


class TTLCache(Generic[V]):
    def __init__(self, name: str, ttl_seconds: float, max_entries: int = 10_000) -> None:
        self.name = name
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: OrderedDict[Any, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> V | None:
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] < time.monotonic():
                if entry is not None:
                    del self._data[key]
                CACHE_MISSES.labels(cache=self.name).inc()
                return None
            self._data.move_to_end(key)
            CACHE_HITS.labels(cache=self.name).inc()
            return entry[1]

    def set(self, key: Any, value: V) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
