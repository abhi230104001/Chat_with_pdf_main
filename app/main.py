"""
app/main.py — FastAPI application factory.

Configures:
- Middleware (request IDs, CORS, static files)
- Startup/shutdown lifecycle
- API route mounting
- Static frontend serving
- Global exception handlers
"""

# from __future__ import annotations

# import time
# import uuid
# from pathlib import Path

# from fastapi import FastAPI, Request, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles

# from app.config import get_settings
# from app.models.schemas import ErrorResponse
# from app.rag.embeddings import get_embedding_dim
# from app.rag.vector_store import init_vector_store
# from app.routes import chat, documents, health
# from app.utils.logging import configure_logging, get_logger, set_request_id

# logger = get_logger(__name__)

# STATIC_DIR = Path(__file__).parent.parent / "static"


# def create_app() -> FastAPI:
#     """Create and configure the FastAPI application."""
#     settings = get_settings()
#     configure_logging(settings.log_level)

#     app = FastAPI(
#         title="Chat with PDF",
#         description="Production-ready RAG chatbot for document Q&A",
#         version="1.0.0",
#         docs_url="/api/docs",
#         redoc_url="/api/redoc",
#         openapi_url="/api/openapi.json",
#     )

#     # ── Middleware ────────────────────────────────────────────────────────────

#     # CORS — restrict in production to your domain
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=False,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     @app.middleware("http")
#     async def request_id_middleware(request: Request, call_next):
#         """Attach a unique request ID to every incoming request."""
#         request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
#         set_request_id(request_id)
#         request.state.request_id = request_id

#         start = time.perf_counter()
#         response = await call_next(request)
#         latency_ms = round((time.perf_counter() - start) * 1000, 1)

#         response.headers["X-Request-ID"] = request_id
#         response.headers["X-Response-Time-Ms"] = str(latency_ms)
#         return response

#     # ── Startup / Shutdown ────────────────────────────────────────────────────

#     @app.on_event("startup")
#     async def on_startup() -> None:
#         logger.info("=== Chat with PDF starting up ===")

#         # Initialize vector store (loads from disk if available)
#         try:
#             dim = get_embedding_dim()
#             init_vector_store(storage_dir=settings.faiss_path, dimension=dim)
#             logger.info("VectorStore initialized (dim=%d, dir=%s)", dim, settings.faiss_dir)
#         except Exception as exc:
#             logger.error("VectorStore initialization failed: %s", exc)
#             # Non-fatal: app can start; uploads will fail until fixed

#         # Ensure upload dir exists
#         settings.upload_path.mkdir(parents=True, exist_ok=True)

#         if not settings.has_api_key():
#             logger.warning(
#                 "GOOGLE_API_KEY is not set. Upload and chat will not work until configured."
#             )
#         else:
#             logger.info("Google API key is configured.")

#         logger.info("=== Startup complete ===")

#     @app.on_event("shutdown")
#     async def on_shutdown() -> None:
#         logger.info("=== Chat with PDF shutting down ===")

#     # ── Exception Handlers ────────────────────────────────────────────────────

#     @app.exception_handler(Exception)
#     async def global_exception_handler(request: Request, exc: Exception):
#         request_id = getattr(request.state, "request_id", "unknown")
#         logger.error("Unhandled exception (request=%s): %s", request_id, exc, exc_info=True)
#         return JSONResponse(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             content=ErrorResponse(
#                 error="An unexpected server error occurred.",
#                 request_id=request_id,
#             ).model_dump(),
#         )

#     # ── API Routes ────────────────────────────────────────────────────────────

#     app.include_router(health.router, tags=["Health"])
#     app.include_router(documents.router, tags=["Documents"])
#     app.include_router(chat.router, tags=["Chat"])

#     # ── Static Files ──────────────────────────────────────────────────────────

#     if STATIC_DIR.exists():
#         app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

#         @app.get("/", include_in_schema=False)
#         async def serve_frontend():
#             index_html = STATIC_DIR / "index.html"
#             if index_html.exists():
#                 return FileResponse(str(index_html))
#             return JSONResponse(
#                 content={"message": "Chat with PDF API", "docs": "/api/docs"},
#                 status_code=200,
#             )
#     else:
#         @app.get("/", include_in_schema=False)
#         async def api_root():
#             return JSONResponse(
#                 content={"message": "Chat with PDF API", "docs": "/api/docs"},
#                 status_code=200,
#             )

#     return app


# # Module-level app instance
# app = create_app()
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.schemas import ErrorResponse
from app.rag.embeddings import get_embedding_dim
from app.rag.vector_store import init_vector_store
from app.routes import chat, documents, health
from app.utils.logging import configure_logging, get_logger, set_request_id

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Chat with PDF",
        description="Production-ready RAG chatbot for document Q&A",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    # CORS — restrict in production to your domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a unique request ID to every incoming request."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        set_request_id(request_id)
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)
        return response

    # ── Startup / Shutdown ───────────────────────────────────────────────────

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("=== Chat with PDF starting up ===")

        # Initialize vector store (loads from disk if available)
        try:
            dim = get_embedding_dim()
            init_vector_store(
                storage_dir=settings.faiss_path,
                dimension=dim,
            )
            logger.info(
                "VectorStore initialized (dim=%d, dir=%s)",
                dim,
                settings.faiss_dir,
            )
        except Exception as exc:
            logger.error("VectorStore initialization failed: %s", exc)
            # Non-fatal: app can start; uploads will fail until fixed

        # Ensure upload directory exists
        settings.upload_path.mkdir(parents=True, exist_ok=True)

        if not settings.has_api_key():
            logger.warning(
                "GOOGLE_API_KEY is not set. "
                "Upload and chat will not work until configured."
            )
        else:
            logger.info("Google API key is configured.")

        logger.info("=== Startup complete ===")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("=== Chat with PDF shutting down ===")

    # ── Exception Handlers ────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")

        logger.error(
            "Unhandled exception (request=%s): %s",
            request_id,
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="An unexpected server error occurred.",
                request_id=request_id,
            ).model_dump(),
        )

    # ── API Routes ────────────────────────────────────────────────────────────

    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(chat.router, tags=["Chat"])

    # ── Static Files ──────────────────────────────────────────────────────────

    if not STATIC_DIR.exists():
        logger.warning("Static directory not found: %s", STATIC_DIR)

    # Serve CSS, JavaScript, and other frontend assets under /static.
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR), check_dir=False),
        name="static",
    )

    # Serve the frontend as the default page at http://localhost:8000/
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index_html = STATIC_DIR / "index.html"

        if index_html.is_file():
            return FileResponse(str(index_html))

        logger.error("Frontend index.html not found: %s", index_html)

        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Frontend file not found",
                "expected_path": str(index_html),
            },
        )

    return app


# Module-level app instance for Uvicorn
app = create_app()
