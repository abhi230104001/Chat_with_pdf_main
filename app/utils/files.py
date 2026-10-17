"""
app/utils/files.py — Filename sanitization, hashing, validation, and safe file I/O.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Allowed file extensions (lowercase)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

# Maximum filename length after sanitization
MAX_FILENAME_LENGTH = 128


def sanitize_filename(filename: str) -> str:
    """
    Return a filesystem-safe version of *filename*.

    Steps:
      1. Normalize Unicode to NFC.
      2. Replace path separators and null bytes.
      3. Remove characters outside the safe ASCII printable set.
      4. Collapse repeated underscores/dashes.
      5. Truncate to MAX_FILENAME_LENGTH (preserving extension).
    """
    # Normalize Unicode
    name = unicodedata.normalize("NFC", filename)

    # Grab extension before sanitizing
    path = Path(name)
    stem = path.stem
    suffix = path.suffix.lower()

    # Remove path traversal attempts
    stem = stem.replace("/", "_").replace("\\", "_").replace("\x00", "")

    # Keep only safe characters
    stem = re.sub(r"[^\w\-. ]", "_", stem)

    # Collapse whitespace and repeated separators
    stem = re.sub(r"[\s_]+", "_", stem).strip("_.-")

    if not stem:
        stem = "document"

    # Truncate
    max_stem = MAX_FILENAME_LENGTH - len(suffix)
    stem = stem[:max_stem]

    safe = f"{stem}{suffix}"
    return safe


def validate_extension(filename: str, allowed: Optional[set] = None) -> bool:
    """Return True if the file's extension is in the allowed set."""
    allowed = allowed or ALLOWED_EXTENSIONS
    ext = Path(filename).suffix.lower()
    return ext in allowed


def compute_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_stream(stream: io.IOBase, chunk_bytes: int = 65536) -> str:
    """Return the hex SHA-256 digest by streaming from *stream*."""
    h = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()


def validate_file_size(size_bytes: int, max_bytes: int) -> bool:
    """Return True if *size_bytes* does not exceed *max_bytes*."""
    return size_bytes <= max_bytes


def safe_delete(path: Path) -> bool:
    """Delete a file safely. Return True if deleted, False if not found."""
    try:
        if path.exists() and path.is_file():
            path.unlink()
            logger.info("Deleted file: %s", path.name)
            return True
        return False
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)
        return False


def get_extension(filename: str) -> str:
    """Return lowercase extension including the dot, e.g. '.pdf'."""
    return Path(filename).suffix.lower()


def human_readable_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"
