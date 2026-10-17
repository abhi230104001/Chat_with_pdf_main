"""
app/routes/chat.py — Chat and streaming chat endpoints.

POST /api/chat        — Standard (blocking) chat.
POST /api/chat/stream — Server-Sent Events streaming chat.
DELETE /api/chat/{session_id} — Clear session history.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DebugInfo,
    DocumentStatus,
)
from app.services.conversation import add_turn, clear_session, get_history_for_prompt
from app.services.generation import generate_answer, stream_answer
from app.services.retrieval import retrieve
from app.utils.logging import get_logger, get_request_id

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


# ── Standard Chat ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with uploaded documents",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask a question about uploaded documents.

    - Retrieves relevant chunks via two-stage retrieval.
    - Generates a grounded answer using Gemini.
    - Returns citations, debug metrics, and the answer.
    - Maintains conversation history per session_id.
    """
    t_total_start = time.perf_counter()

    session_id = request.session_id
    question = request.question

    # Get history before adding current question
    history = get_history_for_prompt(session_id)

    # Retrieval
    try:
        retrieval_result = await retrieve(
            question=question,
            session_id=session_id,
            document_ids=request.document_ids,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    chunks = retrieval_result["chunks"]
    rewritten_query = retrieval_result["rewritten_query"]
    retrieval_latency_ms = retrieval_result["retrieval_latency_ms"]

    # Generation
    try:
        gen_result = await generate_answer(
            question=rewritten_query,
            chunks=chunks,
            history=history,
            retrieval_latency_ms=retrieval_latency_ms,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    answer = gen_result["answer"]
    sources = gen_result["sources"]
    debug: DebugInfo = gen_result["debug"]

    # Update total latency
    total_latency_ms = (time.perf_counter() - t_total_start) * 1000
    debug.total_latency_ms = round(total_latency_ms, 2)
    debug.rewritten_query = rewritten_query

    # Persist conversation turns
    add_turn(session_id, "user", question)
    add_turn(session_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        debug=debug,
        model_used=gen_result.get("model_used"),
    )


# ── Streaming Chat ────────────────────────────────────────────────────────────

@router.post(
    "/chat/stream",
    summary="Stream chat response using Server-Sent Events",
    response_class=StreamingResponse,
)
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """
    Stream the answer using Server-Sent Events (SSE).

    Event types:
      start   — Signals the beginning of generation.
      token   — A text fragment to append.
      sources — Retrieved source citations.
      done    — Final event with latency and debug metrics.
      error   — Error encountered during processing.

    SSE framing: each event is "data: <json>\\n\\n".
    """
    session_id = request.session_id
    question = request.question
    history = get_history_for_prompt(session_id)

    async def event_generator() -> AsyncIterator[str]:
        t_start = time.perf_counter()
        full_answer_parts = []
        final_sources = []
        sent_done = False

        try:
            # Retrieval
            try:
                retrieval_result = await retrieve(
                    question=question,
                    session_id=session_id,
                    document_ids=request.document_ids,
                    history=history,
                )
            except RuntimeError as exc:
                error_data = json.dumps({"event": "error", "data": {"message": str(exc)}})
                yield f"data: {error_data}\n\n"
                return

            chunks = retrieval_result["chunks"]
            rewritten_query = retrieval_result["rewritten_query"]
            retrieval_latency_ms = retrieval_result["retrieval_latency_ms"]

            # Emit start
            start_data = json.dumps({
                "event": "start",
                "data": {
                    "session_id": session_id,
                    "rewritten_query": rewritten_query,
                },
            })
            yield f"data: {start_data}\n\n"

            # Check for client disconnect
            if await http_request.is_disconnected():
                logger.info("Client disconnected before generation (session=%s)", session_id)
                return

            # Stream generation
            async for event in stream_answer(
                question=rewritten_query,
                chunks=chunks,
                history=history,
                retrieval_latency_ms=retrieval_latency_ms,
            ):
                # Check for disconnect on each token
                if await http_request.is_disconnected():
                    logger.info("Client disconnected mid-stream (session=%s)", session_id)
                    return

                event_type = event.get("event")
                event_data = event.get("data", {})

                if event_type == "token":
                    token = event_data.get("token", "")
                    full_answer_parts.append(token)
                    yield f"data: {json.dumps({'event': 'token', 'data': {'token': token}})}\n\n"

                elif event_type == "sources":
                    final_sources = event_data.get("sources", [])
                    yield f"data: {json.dumps({'event': 'sources', 'data': event_data})}\n\n"

                elif event_type == "done":
                    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    event_data["total_latency_ms"] = total_ms
                    yield f"data: {json.dumps({'event': 'done', 'data': event_data})}\n\n"
                    sent_done = True

                    # Save to conversation history
                    full_answer = "".join(full_answer_parts)
                    add_turn(session_id, "user", question)
                    add_turn(session_id, "assistant", full_answer)

                elif event_type == "error":
                    yield f"data: {json.dumps({'event': 'error', 'data': event_data})}\n\n"
                    return

                elif event_type == "start":
                    # Already sent our start event above; skip inner start
                    pass

        except asyncio.CancelledError:
            logger.info("Stream cancelled (session=%s)", session_id)
            return
        except Exception as exc:
            logger.error("Unhandled error in stream (session=%s): %s", session_id, exc)
            if not sent_done:
                error_payload = json.dumps({
                    "event": "error",
                    "data": {"message": "An unexpected error occurred during generation."},
                })
                yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Session Management ────────────────────────────────────────────────────────

@router.delete(
    "/chat/{session_id}",
    summary="Clear conversation history for a session",
)
async def clear_conversation(session_id: str) -> JSONResponse:
    """Clear the in-memory conversation history for the given session."""
    cleared = clear_session(session_id)
    return JSONResponse(
        content={
            "session_id": session_id,
            "cleared": cleared,
            "message": "Session cleared." if cleared else "Session not found.",
        }
    )
