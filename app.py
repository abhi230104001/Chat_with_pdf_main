"""
app.py — Top-level ASGI entry point.

Import target for Uvicorn: `uvicorn app:app`
"""

from app.main import app

__all__ = ["app"]
