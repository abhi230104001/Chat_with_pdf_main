"""
app/routes/documents.py — Document upload, listing, and deletion endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    UploadResponse,
    UploadResult,
)
from app.services.ingestion import delete_document, get_all_documents, get_document, ingest_file
from app.utils.logging import get_logger, get_request_id

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload one or more documents",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more PDF, DOCX, TXT, MD, or CSV files"),
) -> UploadResponse:
    """
    Upload and ingest one or more documents.

    - Validates extensions, sizes, and content.
    - Detects duplicates by SHA-256 hash.
    - Parses, chunks, embeds, and indexes each file.
    - Returns per-file status with chunk counts and any errors.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files provided.",
        )

    results: List[UploadResult] = []
    successful = 0
    failed = 0
    duplicates = 0

    for upload in files:
        filename = upload.filename or "unnamed"
        logger.info("Processing upload: %s (request=%s)", filename, get_request_id())

        try:
            content = await upload.read()
        except Exception as exc:
            logger.error("Failed to read upload %s: %s", filename, exc)
            results.append(UploadResult(
                filename=filename,
                status=DocumentStatus.ERROR,
                error_message="Failed to read uploaded file.",
            ))
            failed += 1
            continue

        result = await ingest_file(content=content, original_filename=filename)
        results.append(result)

        if result.duplicate:
            duplicates += 1
            successful += 1  # Duplicate = already processed = accessible
        elif result.status == DocumentStatus.READY:
            successful += 1
        else:
            failed += 1

    return UploadResponse(
        results=results,
        successful=successful,
        failed=failed,
        duplicates=duplicates,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents() -> DocumentListResponse:
    """Return metadata for all ingested documents."""
    docs = get_all_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document",
)
async def remove_document(document_id: str) -> DeleteResponse:
    """
    Delete a document and all its associated chunks.

    Removes from:
    - Document registry
    - FAISS vector index
    - Upload file storage
    """
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    deleted = await delete_document(document_id)

    if deleted:
        return DeleteResponse(
            document_id=document_id,
            deleted=True,
            message=f"Document '{doc.filename}' deleted successfully.",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document. Please try again.",
        )
