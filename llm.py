"""Language model providers for grounded generation.

``ExtractiveLLM`` needs no external service: it composes an answer from the most relevant
source sentences with citations. It keeps the full RAG path testable offline and acts as a
safe fallback when a hosted model is unavailable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from vantix.config import LLMProvider, Settings
from vantix.http import request_with_retries
from vantix.text import split_sentences, tokenize


@dataclass(slots=True)
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLM(ABC):
    name: str

    @abstractmethod
    def generate(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> LLMResult: ...

    def stream(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> Iterator[str]:
        yield self.generate(system, messages, max_tokens, temperature).text

    def close(self) -> None:  # pragma: no cover
        return None


class OpenAICompatibleLLM(LLM):
    """``POST {base_url}/chat/completions``. Works with OpenAI, Azure gateways, vLLM, Ollama and others."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.model = model
        self.name = model

    def _body(self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float) -> dict:  # type: ignore[type-arg]
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def generate(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> LLMResult:
        response = request_with_retries(
            self._client,
            "POST",
            "/chat/completions",
            json=self._body(system, messages, max_tokens, temperature),
        )
        body = response.json()
        usage = body.get("usage") or {}
        return LLMResult(
            text=body["choices"][0]["message"]["content"] or "",
            model=body.get("model", self.model),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )

    def stream(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> Iterator[str]:
        body = self._body(system, messages, max_tokens, temperature) | {"stream": True}
        with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def close(self) -> None:
        self._client.close()


class AnthropicLLM(LLM):
    """Anthropic Messages API."""

    def __init__(
        self, model: str, api_key: str, base_url: str = "https://api.anthropic.com", timeout: float = 60.0
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
        )
        self.model = model
        self.name = model

    def _body(self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float) -> dict:  # type: ignore[type-arg]
        return {
            "model": self.model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def generate(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> LLMResult:
        response = request_with_retries(
            self._client, "POST", "/v1/messages", json=self._body(system, messages, max_tokens, temperature)
        )
        body = response.json()
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return LLMResult(
            text=text,
            model=body.get("model", self.model),
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )

    def stream(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> Iterator[str]:
        body = self._body(system, messages, max_tokens, temperature) | {"stream": True}
        with self._client.stream("POST", "/v1/messages", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if event.get("type") == "content_block_delta":
                    text = event.get("delta", {}).get("text")
                    if text:
                        yield text

    def close(self) -> None:
        self._client.close()


_SOURCE_RE = re.compile(r"^\[(\d+)\] (.*)$")


class ExtractiveLLM(LLM):
    """Offline grounded answerer: selects the most relevant cited sentences from the sources."""

    name = "extractive"

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(
        self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> LLMResult:
        prompt = messages[-1].content
        question = prompt.rsplit("QUESTION:", 1)[-1].strip()
        q_terms = set(tokenize(question))
        sources = _parse_sources(prompt)
        scored: list[tuple[float, int, int, str]] = []
        for index, text in sources:
            for position, sentence in enumerate(split_sentences(text.replace("\n", " "))):
                terms = set(tokenize(sentence))
                if not terms or not q_terms:
                    continue
                overlap = len(q_terms & terms) / len(q_terms)
                if overlap > 0:
                    # Prefer earlier, denser sentences within higher ranked sources.
                    score = overlap + 0.02 / (1 + position) + 0.01 / index
                    scored.append((score, index, position, sentence.strip()))
        if not scored:
            return LLMResult(text="I could not find this in the available sources.", model=self.name)
        scored.sort(key=lambda x: -x[0])
        chosen: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        # Keep only sentences that are competitive with the best match, so weakly related
        # sentences that share a single common word never pad the answer.
        floor = max(0.3, 0.6 * scored[0][0])
        for score, index, position, sentence in scored:
            if score < floor:
                break
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append((index, position, sentence))
            if len(chosen) >= self.max_sentences:
                break
        chosen.sort(key=lambda x: (x[0], x[1]))
        text = " ".join(f"{s.rstrip()} [{i}]" if not s.endswith(f"[{i}]") else s for i, _, s in chosen)
        return LLMResult(text=text, model=self.name)


def _parse_sources(prompt: str) -> list[tuple[int, str]]:
    block = prompt.split("SOURCES:", 1)[-1].split("QUESTION:", 1)[0]
    sources: list[tuple[int, str]] = []
    current: int | None = None
    lines: list[str] = []
    for line in block.splitlines():
        match = _SOURCE_RE.match(line.strip())
        if match:
            if current is not None:
                sources.append((current, "\n".join(lines)))
            current = int(match.group(1))
            lines = []
        elif current is not None and line.strip() and not line.startswith("</source"):
            lines.append(line)
    if current is not None:
        sources.append((current, "\n".join(lines)))
    return sources


def build_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider is LLMProvider.EXTRACTIVE:
        return ExtractiveLLM()
    if provider is LLMProvider.OPENAI_COMPATIBLE:
        return OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout_seconds
        )
    if provider is LLMProvider.ANTHROPIC:
        base = settings.llm_base_url if "anthropic" in settings.llm_base_url else "https://api.anthropic.com"
        return AnthropicLLM(settings.llm_model, settings.llm_api_key, base, settings.llm_timeout_seconds)
    raise ValueError(f"Unknown LLM provider {provider}")  # pragma: no cover
