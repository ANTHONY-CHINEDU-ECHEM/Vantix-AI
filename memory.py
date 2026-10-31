"""In process indexes for development, testing and small single node deployments.

The vector index performs exact (brute force) inner product search over a contiguous
float32 matrix, which is fast up to a few hundred thousand chunks. The lexical index is a
true incremental Okapi BM25 inverted index. Both are thread safe.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from vantix.domain.models import Chunk
from vantix.index.base import IndexHit, LexicalIndex, SearchFilter, VectorIndex
from vantix.text import tokenize


@dataclass
class _VectorShard:
    dim: int
    matrix: np.ndarray = field(init=False)
    chunks: list[Chunk | None] = field(default_factory=list)
    row_of: dict[str, int] = field(default_factory=dict)
    free_rows: list[int] = field(default_factory=list)
    size: int = 0

    def __post_init__(self) -> None:
        self.matrix = np.zeros((1024, self.dim), dtype=np.float32)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.matrix.shape[0]:
            return
        capacity = self.matrix.shape[0]
        while capacity < needed:
            capacity *= 2
        grown = np.zeros((capacity, self.dim), dtype=np.float32)
        grown[: self.size] = self.matrix[: self.size]
        self.matrix = grown


class MemoryVectorIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._shards: dict[str, _VectorShard] = {}
        self._lock = threading.RLock()

    def _shard(self, tenant_id: str) -> _VectorShard:
        shard = self._shards.get(tenant_id)
        if shard is None:
            shard = _VectorShard(self.dim)
            self._shards[tenant_id] = shard
        return shard

    def upsert(self, tenant_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        with self._lock:
            shard = self._shard(tenant_id)
            for chunk, vector in zip(chunks, vectors, strict=True):
                row = shard.row_of.get(chunk.chunk_id)
                if row is None:
                    if shard.free_rows:
                        row = shard.free_rows.pop()
                    else:
                        shard._ensure_capacity(shard.size + 1)
                        row = shard.size
                        shard.size += 1
                        shard.chunks.append(None)
                    shard.row_of[chunk.chunk_id] = row
                shard.matrix[row] = vector
                shard.chunks[row] = chunk

    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None:
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None:
                return
            for chunk_id in chunk_ids:
                row = shard.row_of.pop(chunk_id, None)
                if row is not None:
                    shard.matrix[row] = 0.0
                    shard.chunks[row] = None
                    shard.free_rows.append(row)

    def search(self, tenant_id: str, vector: np.ndarray, k: int, flt: SearchFilter) -> list[IndexHit]:
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None or shard.size == 0 or k <= 0:
                return []
            scores = shard.matrix[: shard.size] @ np.asarray(vector, dtype=np.float32)
            order = np.argsort(-scores)
            hits: list[IndexHit] = []
            for row in order:
                chunk = shard.chunks[int(row)]
                if chunk is None or not flt.matches(chunk):
                    continue
                hits.append(IndexHit(chunk, float(scores[row])))
                if len(hits) >= k:
                    break
            return hits

    def count(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return 0 if shard is None else len(shard.row_of)

    def drop_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._shards.pop(tenant_id, None)


@dataclass
class _LexicalShard:
    postings: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    lengths: dict[str, int] = field(default_factory=dict)
    terms_of: dict[str, list[str]] = field(default_factory=dict)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    total_length: int = 0


class MemoryBM25Index(LexicalIndex):
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._shards: dict[str, _LexicalShard] = {}
        self._lock = threading.RLock()

    def _shard(self, tenant_id: str) -> _LexicalShard:
        shard = self._shards.get(tenant_id)
        if shard is None:
            shard = _LexicalShard()
            self._shards[tenant_id] = shard
        return shard

    def upsert(self, tenant_id: str, chunks: list[Chunk]) -> None:
        with self._lock:
            shard = self._shard(tenant_id)
            for chunk in chunks:
                self._remove(shard, chunk.chunk_id)
                tokens = tokenize(chunk.text)
                tf: dict[str, int] = defaultdict(int)
                for tok in tokens:
                    tf[tok] += 1
                for term, freq in tf.items():
                    shard.postings[term][chunk.chunk_id] = freq
                shard.lengths[chunk.chunk_id] = len(tokens)
                shard.terms_of[chunk.chunk_id] = list(tf)
                shard.chunks[chunk.chunk_id] = chunk
                shard.total_length += len(tokens)

    def _remove(self, shard: _LexicalShard, chunk_id: str) -> None:
        terms = shard.terms_of.pop(chunk_id, None)
        if terms is None:
            return
        for term in terms:
            posting = shard.postings.get(term)
            if posting is not None:
                posting.pop(chunk_id, None)
                if not posting:
                    del shard.postings[term]
        shard.total_length -= shard.lengths.pop(chunk_id, 0)
        shard.chunks.pop(chunk_id, None)

    def delete(self, tenant_id: str, chunk_ids: list[str]) -> None:
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None:
                return
            for chunk_id in chunk_ids:
                self._remove(shard, chunk_id)

    def search(self, tenant_id: str, query: str, k: int, flt: SearchFilter) -> list[IndexHit]:
        terms = list(dict.fromkeys(tokenize(query)))
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None or not terms or not shard.lengths:
                return []
            n_docs = len(shard.lengths)
            avg_len = shard.total_length / n_docs if n_docs else 1.0
            scores: dict[str, float] = defaultdict(float)
            for term in terms:
                posting = shard.postings.get(term)
                if not posting:
                    continue
                df = len(posting)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                for chunk_id, freq in posting.items():
                    norm = self.k1 * (1.0 - self.b + self.b * shard.lengths[chunk_id] / avg_len)
                    scores[chunk_id] += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            hits: list[IndexHit] = []
            for chunk_id, score in ranked:
                chunk = shard.chunks[chunk_id]
                if not flt.matches(chunk):
                    continue
                hits.append(IndexHit(chunk, score))
                if len(hits) >= k:
                    break
            return hits

    def count(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return 0 if shard is None else len(shard.lengths)

    def drop_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._shards.pop(tenant_id, None)
