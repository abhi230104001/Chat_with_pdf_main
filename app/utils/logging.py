"""
app/utils/logging.py — Structured, Windows-compatible logging with request IDs.

Never logs API keys, full document contents, or sensitive user data.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable to hold per-request IDs
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return request_id_var.get() or "no-request"


def set_request_id(rid: Optional[str] = None) -> str:
    rid = rid or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


class RequestIdFilter(logging.Filter):
    """Injects request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging. Call once at startup."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Remove existing handlers to avoid duplication on hot-reload
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that participates in the configured hierarchy."""
    return logging.getLogger(name)


def scrub_key(value: str, visible: int = 4) -> str:
    """Return a partially redacted API key for safe log messages."""
    if not value or len(value) <= visible:
        return "***"
    return f"{value[:visible]}{'*' * (len(value) - visible)}"
