"""
app/config.py — Centralized application configuration via Pydantic Settings.

Reads from environment variables and .env file. Never hardcodes secrets.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Gemini ─────────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/text-embedding-004"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5
    retrieval_candidates: int = 15  # fetch more, then rerank

    # ── File Upload ───────────────────────────────────────────────────────────
    max_upload_mb: int = 20
    upload_dir: str = "uploads"
    allowed_extensions: List[str] = [".pdf", ".docx", ".txt", ".md", ".csv"]

    # ── Vector Store ─────────────────────────────────────────────────────────
    faiss_dir: str = "faiss_db"

    # ── Server ────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # ── Conversation ─────────────────────────────────────────────────────────
    max_history_turns: int = 10

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def faiss_path(self) -> Path:
        p = Path(self.faiss_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    def has_api_key(self) -> bool:
        return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
