"""
app/rag/chunking.py — Document chunking using LangChain RecursiveCharacterTextSplitter.

Preserves document IDs, filenames, page numbers, chunk indices, and unique chunk IDs.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logging import get_logger

logger = get_logger(__name__)


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a stable, unique chunk ID."""
    return f"{document_id}_chunk_{chunk_index:05d}"


def chunk_document(
    text: str,
    document_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    page_number: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Split a text block into overlapping chunks with full metadata.

    Args:
        text: The raw text content to split.
        document_id: Internal document ID.
        filename: Original (sanitized) filename.
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        page_number: Page number (only for PDF pages, else None).
        extra_metadata: Additional key-value pairs to merge into each chunk's metadata.

    Returns:
        List of chunk dicts: {"chunk_id", "text", "metadata": {...}}.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": idx,
        }

        # Only include page_number when it is meaningful (PDFs)
        if page_number is not None:
            metadata["page_number"] = page_number

        if extra_metadata:
            metadata.update(extra_metadata)

        chunk_id = make_chunk_id(document_id, idx)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": metadata,
        })

    logger.debug(
        "Chunked document %s (page=%s) into %d chunks",
        document_id, page_number, len(chunks),
    )
    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    document_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of page dicts (each with 'text' and optional 'page_number').

    For PDFs this preserves accurate page numbers. Chunk indices are global
    across all pages within the document.

    Args:
        pages: List of {"text": str, "page_number": int | None} dicts.
        document_id: Internal document ID.
        filename: Sanitized filename.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap characters.

    Returns:
        Flat list of chunk dicts with correct page-level metadata.
    """
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 0

    for page in pages:
        page_text = page.get("text", "")
        page_num = page.get("page_number")

        if not page_text or not page_text.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        raw_chunks = splitter.split_text(page_text)

        for chunk_text in raw_chunks:
            if not chunk_text.strip():
                continue

            metadata: Dict[str, Any] = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": global_chunk_idx,
            }
            if page_num is not None:
                metadata["page_number"] = page_num

            chunk_id = make_chunk_id(document_id, global_chunk_idx)

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": metadata,
            })
            global_chunk_idx += 1

    logger.info(
        "Chunked %d pages for doc %s → %d chunks",
        len(pages), document_id, len(all_chunks),
    )
    return all_chunks
