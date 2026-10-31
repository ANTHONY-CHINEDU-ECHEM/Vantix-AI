"""Provider adapters tested against mocked HTTP transports: request shape, parsing and failures."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from vantix.config import Settings
from vantix.embeddings import CachedEmbedder, build_embedder
from vantix.embeddings.hashing import HashingEmbedder
from vantix.embeddings.providers import OpenAICompatibleEmbedder
from vantix.errors import UpstreamError
from vantix.http import request_with_retries
from vantix.rag.llm import AnthropicLLM, ChatMessage, ExtractiveLLM, OpenAICompatibleLLM, build_llm
from vantix.retrieval.rerankers import (
    HttpReranker,
    LexicalReranker,
    NoopReranker,
    build_reranker,
)


def mock_client(handler, base_url: str = "https://upstream.test") -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vantix.http.time.sleep", lambda _seconds: None)


# Retries -------------------------------------------------------------------------------
def test_retries_transient_status_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, headers={"retry-after": "1"})
        return httpx.Response(200, json={"ok": True})

    response = request_with_retries(mock_client(handler), "GET", "/thing")
    assert response.json() == {"ok": True} and calls["n"] == 3


def test_non_retryable_error_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(UpstreamError) as info:
        request_with_retries(mock_client(handler), "POST", "/thing")
    assert calls["n"] == 1 and info.value.details["status"] == 400


def test_transport_errors_exhaust_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(UpstreamError, match="after 3 attempts"):
        request_with_retries(mock_client(handler), "GET", "/thing", max_attempts=3)


# Embeddings ------------------------------------------------------------------------------
def test_openai_compatible_embedder_batches_orders_and_normalises() -> None:
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["input"])
        assert request.url.path == "/v1/embeddings" and body["model"] == "embed_model"
        data = [{"index": i, "embedding": [float(i + 1), 0.0, 0.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    embedder = OpenAICompatibleEmbedder("https://upstream.test/v1", "embed_model", dim=3, batch_size=2)
    embedder._client = mock_client(handler, "https://upstream.test/v1")
    vectors = embedder.embed_documents(["a", "", "c"])
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert seen == [["a", " "], ["c"]], "empty strings are replaced and batches respect batch_size"


def test_embedding_dimension_mismatch_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    embedder = OpenAICompatibleEmbedder("https://upstream.test/v1", "m", dim=3)
    embedder._client = mock_client(handler, "https://upstream.test/v1")
    with pytest.raises(UpstreamError, match="dimension mismatch"):
        embedder.embed_documents(["x"])


def test_cached_embedder_reuses_query_vectors() -> None:
    class Counting(HashingEmbedder):
        calls = 0

        def embed_documents(self, texts: list[str]) -> np.ndarray:
            Counting.calls += 1
            return super().embed_documents(texts)

    cached = CachedEmbedder(Counting(32), max_entries=2)
    first = cached.embed_query("hello")
    assert np.array_equal(first, cached.embed_query("hello"))
    cached.embed_query("a")
    cached.embed_query("b")  # evicts "hello"
    cached.embed_query("hello")
    assert Counting.calls == 4


def test_embedder_factory_selects_provider() -> None:
    assert isinstance(build_embedder(Settings(embedding_dim=16)).inner, HashingEmbedder)  # type: ignore[attr-defined]
    remote = build_embedder(Settings(embedding_provider="openai_compatible", embedding_dim=8))
    assert isinstance(remote.inner, OpenAICompatibleEmbedder)  # type: ignore[attr-defined]


# Language models -------------------------------------------------------------------------
def test_openai_compatible_llm_generate_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        if body.get("stream"):
            chunks = [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(chunks))
        return httpx.Response(
            200,
            json={
                "model": "chat_model",
                "choices": [{"message": {"content": "Answer [1]"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    llm = OpenAICompatibleLLM("https://upstream.test/v1", "chat_model")
    llm._client = mock_client(handler, "https://upstream.test/v1")
    result = llm.generate("system", [ChatMessage("user", "q")], 100, 0.0)
    assert result.text == "Answer [1]" and result.usage == {"input_tokens": 10, "output_tokens": 3}
    assert "".join(llm.stream("system", [ChatMessage("user", "q")], 100, 0.0)) == "Hello"


def test_anthropic_llm_generate_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "secret"
        assert request.headers["anthropic-version"]
        body = json.loads(request.content)
        assert body["system"] == "system" and body["messages"][0]["role"] == "user"
        if body.get("stream"):
            events = [
                'data: {"type":"message_start"}',
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Grounded"}}',
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" [1]"}}',
                'data: {"type":"message_stop"}',
            ]
            return httpx.Response(200, text="\n\n".join(events))
        return httpx.Response(
            200,
            json={
                "model": "claude_model",
                "content": [{"type": "text", "text": "Grounded [1]"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    llm = AnthropicLLM("claude_model", "secret", "https://upstream.test")
    llm._client = mock_client(handler)
    llm._client.headers.update({"x-api-key": "secret", "anthropic-version": "2023-06-01"})
    assert llm.generate("system", [ChatMessage("user", "q")], 50, 0.0).text == "Grounded [1]"
    assert "".join(llm.stream("system", [ChatMessage("user", "q")], 50, 0.0)) == "Grounded [1]"


def test_extractive_llm_handles_no_overlap() -> None:
    prompt = "SOURCES:\n[1] Doc (document)\nCompletely unrelated sentence.\n</source>\n\nQUESTION: zebra"
    result = ExtractiveLLM().generate("s", [ChatMessage("user", prompt)], 100, 0.0)
    assert "could not find" in result.text


def test_llm_factory() -> None:
    assert isinstance(build_llm(Settings()), ExtractiveLLM)
    assert isinstance(build_llm(Settings(llm_provider="openai_compatible")), OpenAICompatibleLLM)
    assert isinstance(build_llm(Settings(llm_provider="anthropic", llm_api_key="k")), AnthropicLLM)


# Rerankers ---------------------------------------------------------------------------------
def test_http_reranker_maps_indexes_and_squashes_logits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "q" and len(body["documents"]) == 3
        return httpx.Response(
            200,
            json={"results": [{"index": 2, "relevance_score": 4.0}, {"index": 0, "relevance_score": -2.0}]},
        )

    reranker = HttpReranker("https://upstream.test/rerank", "rerank_model")
    reranker._client = mock_client(handler)
    scores = reranker.score("q", ["a", "b", "c"])
    assert scores[2] > scores[1] > scores[0] or scores[2] > scores[0]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert reranker.score("q", []) == []


def test_reranker_factory_and_noop() -> None:
    assert isinstance(build_reranker(Settings(reranker_provider="none")), NoopReranker)
    assert isinstance(build_reranker(Settings()), LexicalReranker)
    assert isinstance(
        build_reranker(Settings(reranker_provider="http", reranker_url="https://x")), HttpReranker
    )
    assert NoopReranker().score("q", ["a"]) == [0.0]
    assert LexicalReranker().score("", ["a"]) == [0.0]
