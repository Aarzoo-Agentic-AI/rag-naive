"""Service-level errors, mapped to HTTP status codes in ``app.py``.

Keeping these HTTP-agnostic lets the service layer be reused outside FastAPI.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class; ``status_code`` is the HTTP status the API returns."""

    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class BadRequestError(ServiceError):
    """Invalid input from the client (bad path, not a PDF, too large, ...)."""

    status_code = 400


class NotFoundError(ServiceError):
    """Requested PDF file or folder does not exist."""

    status_code = 404


class LLMNotConfiguredError(ServiceError):
    """GROQ_API_KEY is missing, so question answering is unavailable."""

    status_code = 503
