"""Second stage rerankers. All return scores in [0, 1], aligned to the input order."""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod

import httpx

from vantix.config import RerankerProvider, Settings
from vantix.http import request_with_retries
from vantix.text import tokenize


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...

    def close(self) -> None:  # pragma: no cover
        return None


class NoopReranker(Reranker):
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.0] * len(passages)


class LexicalReranker(Reranker):
    """Dependency free reranker using coverage, phrase and proximity features.

    It is a sensible baseline that consistently beats raw fusion on keyword heavy project
    vocabulary (ticket keys, component names, acronyms). Use a cross encoder for best quality.
    """

    name = "lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms:
            return [0.0] * len(passages)
        q_bigrams = set(zip(q_terms, q_terms[1:], strict=False))
        out: list[float] = []
        for passage in passages:
            tokens = tokenize(passage)
            if not tokens:
                out.append(0.0)
                continue
            positions: dict[str, list[int]] = {}
            for i, tok in enumerate(tokens):
                positions.setdefault(tok, []).append(i)
            matched = [t for t in q_terms if t in positions]
            coverage = len(matched) / len(q_terms)
            bigrams = set(zip(tokens, tokens[1:], strict=False))
            phrase = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            proximity = 0.0
            if len(matched) >= 2:
                firsts = sorted(positions[t][0] for t in matched)
                span = firsts[-1] - firsts[0] + 1
                proximity = len(matched) / span
            header = tokenize(passage.split("\n", 1)[0])
            title_hit = len(set(header) & set(q_terms)) / len(q_terms)
            density = min(1.0, sum(len(positions[t]) for t in matched) / max(8.0, math.sqrt(len(tokens)) * 2))
            score = 0.45 * coverage + 0.2 * phrase + 0.15 * proximity + 0.1 * title_hit + 0.1 * density
            out.append(min(1.0, score))
        return out


class CrossEncoderReranker(Reranker):  # pragma: no cover requires model download
    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            raw = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]


class HttpReranker(Reranker):
    """Calls a ``/rerank`` endpoint using the widely adopted Cohere and Jina request shape."""

    name = "http"

    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 20.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self.url = url
        self.model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        response = request_with_retries(
            self._client,
            "POST",
            self.url,
            json={"model": self.model, "query": query, "documents": passages, "top_n": len(passages)},
        )
        body = response.json()
        results = body.get("results", body if isinstance(body, list) else [])
        scores = [0.0] * len(passages)
        for entry in results:
            value = entry.get("relevance_score", entry.get("score", 0.0))
            scores[int(entry["index"])] = float(value)
        if scores and (max(scores) > 1.0 or min(scores) < 0.0):
            scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        return scores

    def close(self) -> None:
        self._client.close()


def build_reranker(settings: Settings) -> Reranker:
    provider = settings.reranker_provider
    if provider is RerankerProvider.NONE:
        return NoopReranker()
    if provider is RerankerProvider.LEXICAL:
        return LexicalReranker()
    if provider is RerankerProvider.CROSS_ENCODER:  # pragma: no cover
        return CrossEncoderReranker(settings.reranker_model)
    if provider is RerankerProvider.HTTP:
        return HttpReranker(settings.reranker_url, settings.reranker_model, settings.reranker_api_key)
    raise ValueError(f"Unknown reranker {provider}")  # pragma: no cover
