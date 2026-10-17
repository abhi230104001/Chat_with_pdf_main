"""
app/models/schemas.py — Pydantic schemas for the entire application.

Covers requests, responses, events, documents, citations, debug info, and errors.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class StreamEventType(str, Enum):
    START = "start"
    TOKEN = "token"
    SOURCES = "sources"
    DEBUG = "debug"
    DONE = "done"
    ERROR = "error"


# ── Source Citations ──────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A citation pointing to a specific chunk in an uploaded document."""
    chunk_id: str
    document_id: str
    filename: str
    # Page number is only present for PDFs with reliable page metadata
    page_number: Optional[int] = None
    chunk_index: int
    text_snippet: str = Field(..., description="Short excerpt from the chunk")
    # Raw vector L2 distance (lower = more similar for FAISS IndexFlatL2)
    vector_distance: Optional[float] = None
    # Normalized lexical overlap score [0, 1]
    lexical_score: Optional[float] = None
    # Combined reranking score used for final selection
    rerank_score: Optional[float] = None


# ── Chat Schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=128)
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to specific document IDs. None = all documents.",
    )
    stream: bool = False

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class DebugInfo(BaseModel):
    """Real backend metrics returned with every chat response."""
    rewritten_query: Optional[str] = None
    num_candidates_retrieved: int = 0
    num_chunks_selected: int = 0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceCitation] = Field(default_factory=list)
    debug: Optional[DebugInfo] = None
    model_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Streaming Events ──────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    event: StreamEventType
    data: Any = None


class StreamStartData(BaseModel):
    session_id: str
    rewritten_query: Optional[str] = None


class StreamTokenData(BaseModel):
    token: str


class StreamSourcesData(BaseModel):
    sources: List[SourceCitation]


class StreamDoneData(BaseModel):
    session_id: str
    total_latency_ms: float
    sources: List[SourceCitation] = Field(default_factory=list)
    debug: Optional[DebugInfo] = None


class StreamErrorData(BaseModel):
    message: str
    code: Optional[str] = None


# ── Document Schemas ──────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata stored in the document registry for each uploaded file."""
    document_id: str
    filename: str
    original_filename: str
    extension: str
    file_size_bytes: int
    sha256: str
    chunk_count: int
    status: DocumentStatus = DocumentStatus.READY
    error_message: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    # Page count only for PDFs
    page_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int


class UploadResult(BaseModel):
    """Result for a single file in a multi-file upload."""
    filename: str
    document_id: Optional[str] = None
    status: DocumentStatus
    chunk_count: int = 0
    page_count: Optional[int] = None
    error_message: Optional[str] = None
    duplicate: bool = False
    existing_document_id: Optional[str] = None


class UploadResponse(BaseModel):
    results: List[UploadResult]
    successful: int
    failed: int
    duplicates: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    status: str  # "ok" | "degraded" | "error" | "unconfigured"
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "error"
    version: str = "1.0.0"
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
    document_count: int = 0
    vector_count: int = 0
    checked_at: datetime = Field(default_factory=datetime.utcnow)


# ── Error Responses ───────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None
