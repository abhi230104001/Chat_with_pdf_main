"""
app/rag/vector_store.py — FAISS-backed persistent vector store.

Security note: FAISS index files are loaded using faiss.read_index which
deserializes binary data. Only load index files from trusted local paths
that this application controls. Never load user-supplied index files.

Features:
- IndexFlatL2 for exact L2 similarity search (cosine-equivalent after normalization).
- Persistent storage: index + metadata JSON on disk.
- Document-level chunk deletion and index rebuild.
- Safe handling of empty indexes and corrupted persistence.
- Thread-safe via an asyncio.Lock (single-process assumption).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for FAISS to give a clear error if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("faiss-cpu is not installed. Run: pip install faiss-cpu")


class VectorStore:
    """
    FAISS-based vector store with disk persistence.

    Attributes:
        index_path: Path to the FAISS binary index file.
        meta_path: Path to the chunk metadata JSON file.
        dimension: Embedding vector dimension.
    """

    INDEX_FILENAME = "index.faiss"
    META_FILENAME = "metadata.json"

    def __init__(self, storage_dir: Path, dimension: int = 768) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu package is required but not installed.")

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = storage_dir / self.INDEX_FILENAME
        self.meta_path = storage_dir / self.META_FILENAME
        self.dimension = dimension

        # In-memory chunk metadata: position → {chunk_id, document_id, filename, ...}
        self._chunk_meta: List[Dict[str, Any]] = []

        # FAISS index
        self._index: Optional[Any] = None  # faiss.Index

        # Concurrency lock (for async context)
        self._lock = asyncio.Lock()

        self._load_or_create()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        """Load existing index from disk, or create a fresh one."""
        loaded = False

        if self.index_path.exists() and self.meta_path.exists():
            try:
                # Security: only load from our controlled storage directory
                self._index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._chunk_meta = json.load(f)

                if self._index.d != self.dimension:
                    logger.warning(
                        "Stored index dimension %d != configured %d. Rebuilding.",
                        self._index.d, self.dimension,
                    )
                    self._create_fresh_index()
                else:
                    logger.info(
                        "Loaded FAISS index: %d vectors, dim=%d",
                        self._index.ntotal, self.dimension,
                    )
                    loaded = True
            except Exception as exc:
                logger.error("Failed to load FAISS index (%s). Starting fresh.", exc)

        if not loaded:
            self._create_fresh_index()

    def _create_fresh_index(self) -> None:
        """Create a new empty FAISS IndexFlatL2."""
        self._index = faiss.IndexFlatL2(self.dimension)
        self._chunk_meta = []
        logger.info("Created fresh FAISS IndexFlatL2 (dim=%d)", self.dimension)

    def _save(self) -> None:
        """Persist index and metadata to disk atomically where possible."""
        try:
            # Write index
            faiss.write_index(self._index, str(self.index_path))

            # Write metadata atomically via temp file
            tmp_path = self.meta_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._chunk_meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.meta_path)

        except Exception as exc:
            logger.error("Failed to persist FAISS store: %s", exc)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', 'metadata'.
            embeddings: Corresponding embedding vectors (same order as chunks).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )
        if not chunks:
            return

        vectors = np.array(embeddings, dtype=np.float32)

        async with self._lock:
            self._index.add(vectors)
            for chunk, emb in zip(chunks, embeddings):
                meta = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "document_id": chunk["metadata"]["document_id"],
                    "filename": chunk["metadata"]["filename"],
                    "chunk_index": chunk["metadata"].get("chunk_index", 0),
                    "page_number": chunk["metadata"].get("page_number"),
                }
                self._chunk_meta.append(meta)
            self._save()

        logger.info("Added %d chunks to FAISS index (total=%d)", len(chunks), self._index.ntotal)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            document_ids: If provided, restrict results to these document IDs.

        Returns:
            List of result dicts: {chunk_id, text, metadata, vector_distance}.
            Distances are raw L2 values (lower = more similar).
        """
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        # Fetch more candidates if we need to filter by document_ids
        fetch_k = top_k * 4 if document_ids else top_k * 2
        fetch_k = min(fetch_k, self._index.ntotal)

        async with self._lock:
            distances, indices = self._index.search(query, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_meta):
                continue

            meta = self._chunk_meta[idx]

            if document_ids and meta["document_id"] not in document_ids:
                continue

            results.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "metadata": {
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "chunk_index": meta["chunk_index"],
                    "page_number": meta.get("page_number"),
                },
                "vector_distance": float(dist),
            })

            if len(results) >= top_k:
                break

        return results

    async def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks associated with *document_id* and rebuild the index.

        Rebuilding is necessary because FAISS IndexFlatL2 does not support
        in-place deletion. This is O(n) but acceptable for typical corpus sizes.

        Returns:
            Number of chunks removed.
        """
        async with self._lock:
            original_count = len(self._chunk_meta)
            keep_meta = [m for m in self._chunk_meta if m["document_id"] != document_id]
            removed = original_count - len(keep_meta)

            if removed == 0:
                logger.info("No chunks found for document %s", document_id)
                return 0

            # Rebuild index with remaining chunks
            # We need to recover the original vectors — but IndexFlatL2 stores them
            if hasattr(self._index, "reconstruct"):
                keep_indices = [
                    i for i, m in enumerate(self._chunk_meta)
                    if m["document_id"] != document_id
                ]
                if keep_indices:
                    vectors = np.array(
                        [self._index.reconstruct(i) for i in keep_indices],
                        dtype=np.float32,
                    )
                    new_index = faiss.IndexFlatL2(self.dimension)
                    new_index.add(vectors)
                else:
                    new_index = faiss.IndexFlatL2(self.dimension)
            else:
                new_index = faiss.IndexFlatL2(self.dimension)

            self._index = new_index
            self._chunk_meta = keep_meta
            self._save()

        logger.info(
            "Deleted %d chunks for document %s (index rebuilt, %d remain)",
            removed, document_id, len(keep_meta),
        )
        return removed

    def get_vector_count(self) -> int:
        """Return the current number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    def get_document_chunk_count(self, document_id: str) -> int:
        """Return the number of indexed chunks for a specific document."""
        return sum(1 for m in self._chunk_meta if m["document_id"] == document_id)

    def is_empty(self) -> bool:
        """Return True if the index contains no vectors."""
        return self.get_vector_count() == 0


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the module-level VectorStore singleton."""
    global _store
    if _store is None:
        raise RuntimeError("VectorStore has not been initialized. Call init_vector_store() first.")
    return _store


def init_vector_store(storage_dir: Path, dimension: int = 768) -> VectorStore:
    """Initialize (or reinitialize) the module-level VectorStore."""
    global _store
    _store = VectorStore(storage_dir=storage_dir, dimension=dimension)
    return _store
