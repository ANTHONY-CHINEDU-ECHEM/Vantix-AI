"""Index factories."""

from __future__ import annotations

from vantix.config import LexicalBackend, Settings, VectorBackend
from vantix.index.base import IndexHit, LexicalIndex, SearchFilter, VectorIndex
from vantix.index.memory import MemoryBM25Index, MemoryVectorIndex


def build_vector_index(settings: Settings, dim: int) -> VectorIndex:
    if settings.vector_backend is VectorBackend.MEMORY:
        return MemoryVectorIndex(dim)
    from vantix.index.qdrant import QdrantVectorIndex

    return QdrantVectorIndex(
        settings.qdrant_url, settings.qdrant_collection, dim, settings.qdrant_api_key, settings.qdrant_hnsw_ef
    )


def build_lexical_index(settings: Settings) -> LexicalIndex:
    if settings.lexical_backend is LexicalBackend.MEMORY:
        return MemoryBM25Index()
    from vantix.index.opensearch import OpenSearchLexicalIndex

    return OpenSearchLexicalIndex(
        settings.opensearch_url,
        settings.opensearch_index,
        settings.opensearch_username,
        settings.opensearch_password,
    )


__all__ = [
    "IndexHit",
    "LexicalIndex",
    "MemoryBM25Index",
    "MemoryVectorIndex",
    "SearchFilter",
    "VectorIndex",
    "build_lexical_index",
    "build_vector_index",
]
