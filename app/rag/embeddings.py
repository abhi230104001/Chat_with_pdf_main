"""
app/rag/embeddings.py — Google Generative AI embeddings using the google-genai SDK v2.

Uses google.genai (google-genai package >=2.0) which is the current maintained SDK.
Supports configurable models, exponential backoff, batching, and dimension validation.
"""


from __future__ import annotations

import logging
from typing import List

from google.genai import types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Known embedding dimensions per model
KNOWN_DIMENSIONS: dict[str, int] = {
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
    "text-embedding-004": 768,
    "gemini-embedding-001": 768,
    "models/gemini-embedding-001": 768,
}

_client = None


def _get_client():
    """Return a cached google.genai.Client instance."""
    global _client

    if _client is not None:
        return _client

    settings = get_settings()

    if not settings.has_api_key():
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Set it in your .env file before using embeddings."
        )

    try:
        from google import genai

        _client = genai.Client(api_key=settings.google_api_key)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Google AI client: {exc}"
        ) from exc

    return _client


def get_embedding_dim() -> int:
    """Return the expected embedding dimension for the configured model."""
    settings = get_settings()
    model = settings.embedding_model

    return KNOWN_DIMENSIONS.get(model, KNOWN_DIMENSIONS.get(
        model.removeprefix("models/"), 768
    ))


def _normalize_model_name(model: str) -> str:
    """Ensure model name uses the 'models/' prefix."""
    if not model.startswith("models/"):
        return f"models/{model}"

    return model


def _embed_with_dimension(
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """Generate embeddings with the dimension required by FAISS."""
    client = _get_client()
    model_name = _normalize_model_name(model)
    expected_dim = get_embedding_dim()

    result = client.models.embed_content(
        model=model_name,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=expected_dim
        ),
    )

    embeddings = [
        list(embedding.values)
        for embedding in (result.embeddings or [])
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, received {len(embeddings)}."
        )

    for idx, embedding in enumerate(embeddings):
        if len(embedding) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch at index {idx}: "
                f"received {len(embedding)}, expected {expected_dim}."
            )

    return embeddings


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed_batch(texts: List[str], model: str) -> List[List[float]]:
    """Embed a batch of document texts with retry logic."""
    return _embed_with_dimension(texts, model)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed_query_single(text: str, model: str) -> List[float]:
    """Embed a single query text with retry logic."""
    embeddings = _embed_with_dimension([text], model)
    return embeddings[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate document embeddings for a list of texts.

    Batches calls to avoid oversized API requests.

    Raises:
        RuntimeError: If the API key is missing or embedding dimensions
                      are inconsistent.
        Exception: After retry exhaustion on API errors.
    """
    if not texts:
        return []

    settings = get_settings()
    model = settings.embedding_model

    batch_size = 20
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        logger.debug(
            "Embedding batch %d-%d (model=%s)",
            i,
            i + len(batch),
            model,
        )

        batch_embeddings = _embed_batch(batch, model)
        all_embeddings.extend(batch_embeddings)

    expected_dim = get_embedding_dim()

    logger.info(
        "Generated %d embeddings (dim=%d)",
        len(all_embeddings),
        expected_dim,
    )

    return all_embeddings


def embed_query(text: str) -> List[float]:
    """
    Generate a query embedding for retrieval.

    Raises:
        RuntimeError: If the API key is missing or the embedding dimension
                      does not match the configured dimension.
        Exception: After retry exhaustion on API errors.
    """
    settings = get_settings()
    model = settings.embedding_model

    logger.debug(
        "Embedding query (model=%s, len=%d)",
        model,
        len(text),
    )

    embedding = _embed_query_single(text, model)

    logger.debug(
        "Generated query embedding (dim=%d)",
        len(embedding),
    )

    return embedding
