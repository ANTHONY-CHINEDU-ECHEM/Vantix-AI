"""Authentication, tenant resolution, rate limiting and webhook signatures."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass

from vantix.errors import AuthenticationError, PermissionDenied, RateLimited


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    tenant_id: str
    is_admin: bool
    key_fingerprint: str


class ApiKeyAuthenticator:
    """Maps bearer API keys to tenants. Keys are compared in constant time."""

    def __init__(self, tenant_keys: dict[str, str], admin_keys: dict[str, str]) -> None:
        self._tenant = tenant_keys
        self._admin = admin_keys

    def authenticate(self, authorization: str | None) -> CallerIdentity:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthenticationError("Missing bearer token")
        token = authorization[7:].strip()
        for table, is_admin in ((self._admin, True), (self._tenant, False)):
            for key, tenant in table.items():
                if hmac.compare_digest(key.encode(), token.encode()):
                    return CallerIdentity(tenant, is_admin, hashlib.sha256(token.encode()).hexdigest()[:12])
        raise AuthenticationError("Invalid API key")

    @staticmethod
    def require_admin(identity: CallerIdentity) -> None:
        if not identity.is_admin:
            raise PermissionDenied("Administrative key required")


class RateLimiter:
    """Token bucket per API key. Replace with a shared limiter (gateway or Redis) when scaling out."""

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> None:
        if self.capacity <= 0:
            return
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.rate)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                raise RateLimited(
                    "Rate limit exceeded",
                    details={"retry_after_seconds": round((cost - tokens) / self.rate, 2)},
                )
            self._buckets[key] = (tokens - cost, now)


def sign_payload(secret: str, body: bytes, timestamp: int | None = None) -> str:
    """Produce the value of the ``Signature`` header: ``t=<unix>,v1=<hex hmac sha256>``."""
    ts = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str | None, tolerance_seconds: int = 300) -> None:
    if not header:
        raise AuthenticationError("Missing Signature header")
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    try:
        ts = int(parts["t"])
        provided = parts["v1"]
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Malformed Signature header") from exc
    if abs(time.time() - ts) > tolerance_seconds:
        raise AuthenticationError("Signature timestamp outside tolerance")
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, provided):
        raise AuthenticationError("Invalid signature")
