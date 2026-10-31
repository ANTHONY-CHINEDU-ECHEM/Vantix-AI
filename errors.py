"""Typed errors mapped to stable HTTP responses by the API layer."""

from __future__ import annotations


class VantixError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(VantixError):
    status_code = 404
    code = "not_found"


class ValidationFailed(VantixError):
    status_code = 422
    code = "validation_failed"


class AuthenticationError(VantixError):
    status_code = 401
    code = "unauthenticated"


class PermissionDenied(VantixError):
    status_code = 403
    code = "forbidden"


class RateLimited(VantixError):
    status_code = 429
    code = "rate_limited"


class QueueFull(VantixError):
    status_code = 503
    code = "queue_full"


class UpstreamError(VantixError):
    status_code = 502
    code = "upstream_error"
