"""
app/services/retrieval.py — Two-stage retrieval with reranking.

Stage 1: Vector search (FAISS) with optional query rewriting.
Stage 2: Lexical reranking via normalized token overlap, deduplication,
         and document filtering.

Scores returned:
  - vector_distance: Raw FAISS L2 distance (lower = more similar).
  - lexical_score:   Normalized token overlap with query [0, 1].
  - rerank_score:    Combined score for final ranking (higher = better).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.prompts import build_rewrite_prompt
from app.rag.vector_store import get_vector_store
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Query Rewriting ───────────────────────────────────────────────────────────

async def rewrite_query(
    question: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Rewrite a follow-up question into a standalone search query.

    Uses the Gemini generation model to resolve references against history.
    Falls back to the original question on any error.
    """
    if not history:
        return question

    from app.config import get_settings
    settings = get_settings()

    if not settings.has_api_key():
        return question

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        prompt = build_rewrite_prompt(question, history)
        response = model.generate_content(prompt)
        rewritten = response.text.strip()

        if rewritten and len(rewritten) < 500:
            logger.debug("Query rewritten: %r → %r", question, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (%s), using original", exc)

    return question


# ── Lexical Scoring ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Return a set of lowercase, non-empty tokens from text."""
    return {w.lower().strip(".,!?;:\"'()[]") for w in text.split() if w.strip()}


def _lexical_score(query: str, chunk_text: str) -> float:
    """
    Compute normalized token overlap between query and chunk.

    score = |intersection| / |query_tokens|
    Returns 0.0 if query has no tokens.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(chunk_text)
    overlap = q_tokens & c_tokens
    return len(overlap) / len(q_tokens)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(chunks: List[Dict[str, Any]], min_diff_ratio: float = 0.15) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks based on token Jaccard similarity.

    Two chunks are considered duplicates if their Jaccard similarity exceeds
    (1 - min_diff_ratio).
    """
    seen_tokens: List[set] = []
    unique = []
    threshold = 1.0 - min_diff_ratio

    for chunk in chunks:
        tokens = _tokenize(chunk.get("text", ""))
        is_dup = False
        for prev_tokens in seen_tokens:
            if not (tokens | prev_tokens):
                continue
            jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen_tokens.append(tokens)
            unique.append(chunk)

    return unique


# ── Combined Reranking Score ──────────────────────────────────────────────────

def _combine_scores(vector_distance: float, lexical_score: float) -> float:
    """
    Combine FAISS L2 distance and lexical overlap into a single rerank score.

    Strategy:
      - Convert L2 distance to a pseudo-similarity: sim = 1 / (1 + distance).
        This maps [0, ∞) → (0, 1] monotonically.
      - Weighted sum: 0.7 * vector_sim + 0.3 * lexical_score.

    The 0.7/0.3 split is a pragmatic default. It is documented here, not
    presented as a calibrated probability.
    """
    vector_sim = 1.0 / (1.0 + vector_distance)
    return 0.7 * vector_sim + 0.3 * lexical_score


# ── Main Retrieval Function ───────────────────────────────────────────────────

async def retrieve(
    question: str,
    session_id: str = "default",
    document_ids: Optional[List[str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Execute two-stage retrieval for *question*.

    Returns a dict with:
      rewritten_query: The (possibly rewritten) search query.
      chunks: Final selected chunks with scores.
      retrieval_latency_ms: Time for FAISS search + embedding.
      num_candidates: Number of candidates fetched from FAISS.
      num_selected: Number of chunks after reranking.
    """
    settings = get_settings()
    history = history or []

    t_start = time.perf_counter()

    # Stage 1a: Query rewriting
    rewritten_query = await rewrite_query(question, history)

    # Stage 1b: Embed query
    try:
        q_embedding = embed_query(rewritten_query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise RuntimeError(f"Failed to generate query embedding: {exc}") from exc

    # Stage 1c: FAISS candidate retrieval
    store = get_vector_store()
    candidates = await store.search(
        query_embedding=q_embedding,
        top_k=settings.retrieval_candidates,
        document_ids=document_ids,
    )

    t_retrieved = time.perf_counter()
    retrieval_latency_ms = (t_retrieved - t_start) * 1000

    if not candidates:
        logger.info("No candidates retrieved for query: %r", rewritten_query[:80])
        return {
            "rewritten_query": rewritten_query,
            "chunks": [],
            "retrieval_latency_ms": retrieval_latency_ms,
            "num_candidates": 0,
            "num_selected": 0,
        }

    # Stage 2a: Lexical scoring
    for candidate in candidates:
        lex_score = _lexical_score(rewritten_query, candidate.get("text", ""))
        v_dist = candidate.get("vector_distance", 1e6)
        rerank = _combine_scores(v_dist, lex_score)
        candidate["lexical_score"] = round(lex_score, 4)
        candidate["rerank_score"] = round(rerank, 4)

    # Stage 2b: Sort by combined score descending
    candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

    # Stage 2c: Deduplicate
    unique_candidates = _deduplicate(candidates)

    # Stage 2d: Select final top-k
    selected = unique_candidates[: settings.top_k]

    logger.info(
        "Retrieved %d candidates → %d unique → %d selected for query: %r",
        len(candidates), len(unique_candidates), len(selected), rewritten_query[:60],
    )

    return {
        "rewritten_query": rewritten_query,
        "chunks": selected,
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "num_candidates": len(candidates),
        "num_selected": len(selected),
    }
