"""Shared HTTP client helpers with bounded retries and exponential backoff."""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

from vantix.errors import UpstreamError
from vantix.observability.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    **kwargs: Any,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                delay = _retry_after(response) or _backoff(attempt, base_delay, max_delay)
                log.warning(
                    "upstream_retry", extra={"url": url, "status": response.status_code, "delay": delay}
                )
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                raise UpstreamError(
                    f"{method} {url} failed with status {response.status_code}",
                    details={"status": response.status_code, "body": response.text[:500]},
                )
            return response
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = _backoff(attempt, base_delay, max_delay)
            log.warning("upstream_transport_retry", extra={"url": url, "error": str(exc), "delay": delay})
            time.sleep(delay)
    raise UpstreamError(f"{method} {url} failed after {max_attempts} attempts: {last_error}")


def _backoff(attempt: int, base: float, cap: float) -> float:
    # jitter, not cryptography
    return min(cap, base * (2 ** (attempt - 1))) * (0.5 + random.random() / 2)  # nosec B311


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return min(30.0, float(value))
    except ValueError:
        return None
