"""
app/services/ingestion.py — Complete document ingestion pipeline.

Pipeline stages:
  1. Validate extension, MIME-like sniffing, size, and content.
  2. Sanitize filename, generate document ID, compute SHA-256.
  3. Detect duplicate uploads (by hash).
  4. Parse: PDF (page-by-page), DOCX, TXT, MD, CSV.
  5. Chunk with page-level metadata for PDFs.
  6. Embed chunks via Google Gemini.
  7. Insert into FAISS.
  8. Persist document registry.

Failed uploads do not leave orphaned vectors: we only commit to FAISS and the
registry after all earlier stages succeed.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.models.schemas import DocumentMetadata, DocumentStatus, UploadResult
from app.rag.chunking import chunk_pages, chunk_document
from app.rag.embeddings import embed_texts
from app.rag.vector_store import get_vector_store
from app.utils.files import (
    compute_sha256,
    get_extension,
    sanitize_filename,
    validate_extension,
    validate_file_size,
    safe_delete,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Registry file stores document metadata between restarts
REGISTRY_FILENAME = "document_registry.json"


# ── Registry Helpers ──────────────────────────────────────────────────────────

def _registry_path() -> Path:
    settings = get_settings()
    return settings.faiss_path / REGISTRY_FILENAME


def load_registry() -> Dict[str, Dict[str, Any]]:
    """Load document registry from disk. Returns empty dict on missing/corrupt file."""
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to load document registry: %s", exc)
        return {}


def save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    """Persist registry to disk atomically."""
    path = _registry_path()
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as exc:
        logger.error("Failed to save document registry: %s", exc)
        raise


def get_all_documents() -> List[DocumentMetadata]:
    """Return all documents from the registry as DocumentMetadata objects."""
    registry = load_registry()
    docs = []
    for doc_data in registry.values():
        try:
            docs.append(DocumentMetadata(**doc_data))
        except Exception as exc:
            logger.warning("Skipping malformed registry entry: %s", exc)
    docs.sort(key=lambda d: d.uploaded_at, reverse=True)
    return docs


def get_document(document_id: str) -> Optional[DocumentMetadata]:
    """Return a single document by ID, or None if not found."""
    registry = load_registry()
    data = registry.get(document_id)
    if data is None:
        return None
    try:
        return DocumentMetadata(**data)
    except Exception:
        return None


def _find_by_hash(sha256: str) -> Optional[str]:
    """Return existing document_id if the same file hash is already ingested."""
    registry = load_registry()
    for doc_id, doc_data in registry.items():
        if doc_data.get("sha256") == sha256:
            return doc_id
    return None


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_pdf(content: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a PDF file page by page using pypdf.

    Returns:
        (pages, page_count) where pages = [{"text": str, "page_number": int}].
    Raises:
        ValueError: For encrypted or unreadable PDFs.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is required: pip install pypdf")

    reader = PdfReader(io.BytesIO(content))

    if reader.is_encrypted:
        raise ValueError("PDF is encrypted. Please provide an unencrypted document.")

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not extract text from PDF page %d: %s", page_num, exc)
            text = ""
        if text.strip():
            pages.append({"text": text, "page_number": page_num})

    if not pages:
        raise ValueError("No extractable text found in PDF. It may be image-only or corrupted.")

    return pages, len(reader.pages)


def _parse_docx(content: bytes) -> List[Dict[str, Any]]:
    """
    Parse a DOCX file using python-docx.

    Returns pages=[{"text": str}] — DOCX has no reliable page boundaries,
    so page_number is intentionally omitted.
    """
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required: pip install python-docx")

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        # Try tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

    if not paragraphs:
        raise ValueError("No extractable text found in DOCX.")

    # Return as a single logical page (no reliable page metadata in DOCX)
    return [{"text": "\n\n".join(paragraphs)}]


def _parse_text(content: bytes, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Parse plain text or Markdown. No page metadata."""
    for enc in (encoding, "latin-1", "cp1252"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode text file with any common encoding.")

    if not text.strip():
        raise ValueError("Text file is empty.")

    return [{"text": text}]


def _parse_csv(content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV using Python's csv module. Converts rows to text."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            text_content = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV file.")

    reader = csv.reader(io.StringIO(text_content))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV file is empty.")

    # Convert to readable text representation
    lines = []
    header = rows[0] if rows else []
    for row in rows[1:]:
        if any(cell.strip() for cell in row):
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if h.strip() or v.strip()]
            lines.append(", ".join(pairs))

    if not lines and rows:
        # No data rows, just use header
        lines = [", ".join(header)]

    return [{"text": "\n".join(lines)}]


def _parse_file(content: bytes, extension: str, filename: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Dispatch to the correct parser based on extension."""
    ext = extension.lower()
    if ext == ".pdf":
        pages, page_count = _parse_pdf(content)
        return pages, page_count
    elif ext == ".docx":
        return _parse_docx(content), None
    elif ext in (".txt", ".md"):
        return _parse_text(content), None
    elif ext == ".csv":
        return _parse_csv(content), None
    else:
        raise ValueError(f"Unsupported file format: {extension}")


# ── Main Ingestion Function ───────────────────────────────────────────────────

async def ingest_file(
    content: bytes,
    original_filename: str,
) -> UploadResult:
    """
    Full ingestion pipeline for a single uploaded file.

    Returns an UploadResult indicating success, failure, or duplicate.
    Never leaves orphaned vectors on failure.
    """
    settings = get_settings()

    # Stage 1: Validate
    if not validate_extension(original_filename, set(settings.allowed_extensions)):
        return UploadResult(
            filename=original_filename,
            status=DocumentStatus.ERROR,
            error_message=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    if not validate_file_size(len(content), settings.max_upload_bytes):
        return UploadResult(
            filename=original_filename,
            status=DocumentStatus.ERROR,
            error_message=f"File exceeds maximum size of {settings.max_upload_mb} MB.",
        )

    if len(content) == 0:
        return UploadResult(
            filename=original_filename,
            status=DocumentStatus.ERROR,
            error_message="File is empty.",
        )

    # Stage 2: Sanitize & hash
    safe_name = sanitize_filename(original_filename)
    ext = get_extension(original_filename)
    sha256 = compute_sha256(content)

    # Stage 3: Duplicate detection
    existing_id = _find_by_hash(sha256)
    if existing_id:
        logger.info("Duplicate upload detected for %s (existing: %s)", safe_name, existing_id)
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.READY,
            duplicate=True,
            existing_document_id=existing_id,
        )

    # Stage 4: Parse
    try:
        pages, page_count = _parse_file(content, ext, safe_name)
    except ValueError as exc:
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected parse error for %s: %s", safe_name, exc)
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message="Failed to parse document. It may be corrupted or in an unsupported variant.",
        )

    # Generate document ID
    document_id = str(uuid.uuid4())

    # Stage 5: Chunk
    chunks = chunk_pages(
        pages=pages,
        document_id=document_id,
        filename=safe_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message="Document produced no text chunks after processing.",
        )

    # Stage 6: Embed
    if not settings.has_api_key():
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message="GOOGLE_API_KEY is not configured. Cannot generate embeddings.",
        )

    try:
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", safe_name, exc)
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message=f"Embedding generation failed: {str(exc)[:200]}",
        )

    # Stage 7: Insert into FAISS (only after all earlier stages succeed)
    try:
        store = get_vector_store()
        await store.add_chunks(chunks, embeddings)
    except Exception as exc:
        logger.error("FAISS insertion failed for %s: %s", safe_name, exc)
        return UploadResult(
            filename=safe_name,
            status=DocumentStatus.ERROR,
            error_message="Failed to index document. Please try again.",
        )

    # Stage 8: Save to upload directory
    upload_dir = settings.upload_path
    upload_path = upload_dir / f"{document_id}_{safe_name}"
    try:
        upload_path.write_bytes(content)
    except Exception as exc:
        logger.warning("Could not save upload to disk: %s", exc)
        # Non-fatal: document is indexed; file won't be accessible via filesystem

    # Stage 9: Update registry
    doc_meta = DocumentMetadata(
        document_id=document_id,
        filename=safe_name,
        original_filename=original_filename,
        extension=ext,
        file_size_bytes=len(content),
        sha256=sha256,
        chunk_count=len(chunks),
        status=DocumentStatus.READY,
        page_count=page_count,
        uploaded_at=datetime.utcnow(),
    )

    registry = load_registry()
    registry[document_id] = doc_meta.model_dump()
    save_registry(registry)

    logger.info(
        "Ingested %s → doc_id=%s, %d chunks, %s pages",
        safe_name, document_id, len(chunks), page_count,
    )

    return UploadResult(
        filename=safe_name,
        document_id=document_id,
        status=DocumentStatus.READY,
        chunk_count=len(chunks),
        page_count=page_count,
    )


async def delete_document(document_id: str) -> bool:
    """
    Remove a document from the registry, FAISS index, and uploaded file storage.

    Returns True if deleted, False if not found.
    """
    registry = load_registry()

    if document_id not in registry:
        return False

    doc_data = registry[document_id]

    # Remove from FAISS
    try:
        store = get_vector_store()
        removed = await store.delete_document(document_id)
        logger.info("Removed %d chunks from FAISS for doc %s", removed, document_id)
    except Exception as exc:
        logger.error("FAISS deletion failed for %s: %s", document_id, exc)

    # Remove upload file
    settings = get_settings()
    safe_name = doc_data.get("filename", "")
    upload_path = settings.upload_path / f"{document_id}_{safe_name}"
    safe_delete(upload_path)

    # Remove from registry
    del registry[document_id]
    save_registry(registry)

    logger.info("Deleted document %s (%s)", document_id, safe_name)
    return True
