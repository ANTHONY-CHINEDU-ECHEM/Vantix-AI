"""Prometheus metrics. Exposed at ``/v1/metrics``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

HTTP_REQUESTS = Counter(
    "vantix_http_requests_total",
    "HTTP requests by route and status",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "vantix_http_request_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
STAGE_LATENCY = Histogram(
    "vantix_stage_seconds",
    "Latency of internal pipeline stages",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
ITEMS_INDEXED = Counter("vantix_items_indexed_total", "Items indexed", ["tenant"], registry=REGISTRY)
CHUNKS_INDEXED = Counter("vantix_chunks_indexed_total", "Chunks indexed", ["tenant"], registry=REGISTRY)
EVENTS_RECEIVED = Counter("vantix_events_total", "Events received", ["type", "outcome"], registry=REGISTRY)
CACHE_HITS = Counter("vantix_cache_hits_total", "Cache hits", ["cache"], registry=REGISTRY)
CACHE_MISSES = Counter("vantix_cache_misses_total", "Cache misses", ["cache"], registry=REGISTRY)
QUEUE_DEPTH = Gauge("vantix_ingestion_queue_depth", "Pending ingestion jobs", registry=REGISTRY)
RAG_ABSTENTIONS = Counter(
    "vantix_rag_abstentions_total", "Questions answered with an abstention", registry=REGISTRY
)
MODEL_TRAININGS = Counter(
    "vantix_model_trainings_total", "Recommender training runs", ["tenant", "outcome"], registry=REGISTRY
)


class Timer:
    """Context manager recording a stage duration."""

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._ctx = STAGE_LATENCY.labels(stage=stage).time()

    def __enter__(self) -> Timer:
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(*exc)  # type: ignore[arg-type]
