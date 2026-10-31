"""Deterministic feature hashing embedder.

Requires no model download and no network, which makes it ideal for development, CI,
air gapped evaluation and as an emergency fallback. It captures lexical and sub word
similarity (unigrams, bigrams and character trigrams) but not deep semantics, so production
deployments should configure a neural provider.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from vantix.embeddings.base import Embedder, l2_normalise
from vantix.text import tokenize


@lru_cache(maxsize=500_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing_{dim}"

    def _features(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        feats: dict[str, float] = {}
        for tok in tokens:
            feats["u:" + tok] = feats.get("u:" + tok, 0.0) + 1.0
            if len(tok) > 3:
                padded = f"#{tok}#"
                for i in range(len(padded) - 2):
                    key = "c:" + padded[i : i + 3]
                    feats[key] = feats.get(key, 0.0) + 0.25
        for a, b in zip(tokens, tokens[1:], strict=False):
            key = f"b:{a}_{b}"
            feats[key] = feats.get(key, 0.0) + 0.75
        return feats

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                index, sign = _bucket(feature, self.dim)
                out[row, index] += sign * (1.0 + np.log1p(count))
        return l2_normalise(out)
