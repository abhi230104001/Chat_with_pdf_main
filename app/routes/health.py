"""
app/routes/health.py — Health check endpoint.

Reports application and component status without exposing secrets or
sensitive filesystem details.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import ComponentHealth, HealthResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Application health check")
async def health_check() -> HealthResponse:
    """
    Return the health status of application components.

    Never exposes API keys or sensitive filesystem paths.
    """
    settings = get_settings()
    components: dict = {}

    # API key configured?
    if settings.has_api_key():
        components["gemini_api"] = ComponentHealth(status="ok")
    else:
        components["gemini_api"] = ComponentHealth(
            status="unconfigured",
            detail="GOOGLE_API_KEY is not set. Set it in your .env file.",
        )

    # Vector store
    doc_count = 0
    vector_count = 0
    try:
        from app.rag.vector_store import get_vector_store
        store = get_vector_store()
        vector_count = store.get_vector_count()
        components["vector_store"] = ComponentHealth(
            status="ok",
            detail=f"{vector_count} vectors indexed",
        )
    except RuntimeError:
        components["vector_store"] = ComponentHealth(
            status="error",
            detail="VectorStore not initialized",
        )
    except Exception as exc:
        components["vector_store"] = ComponentHealth(
            status="error",
            detail=str(exc)[:100],
        )

    # Document registry
    try:
        from app.services.ingestion import get_all_documents
        docs = get_all_documents()
        doc_count = len(docs)
        components["document_registry"] = ComponentHealth(
            status="ok",
            detail=f"{doc_count} documents",
        )
    except Exception as exc:
        components["document_registry"] = ComponentHealth(
            status="error",
            detail=str(exc)[:100],
        )

    # Overall status
    statuses = [c.status for c in components.values()]
    if "error" in statuses:
        overall = "degraded"
    elif "unconfigured" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        components=components,
        document_count=doc_count,
        vector_count=vector_count,
    )
