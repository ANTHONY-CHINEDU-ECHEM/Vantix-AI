"""Structure aware chunking.

Documents are split along Markdown headings first, then paragraphs, then sentences, and
packed into chunks close to a target token size with a sliding overlap. Every chunk is
prefixed with a short contextual header (item title and section heading) because that
header measurably improves both dense and lexical retrieval on short, ambiguous passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vantix.text import count_tokens, split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(slots=True)
class TextChunk:
    ordinal: int
    text: str
    heading: str | None
    token_count: int


@dataclass(slots=True)
class _Section:
    heading: str | None
    text: str


class Chunker:
    def __init__(self, target_tokens: int = 350, overlap_tokens: int = 50, min_tokens: int = 40) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be in [0, target_tokens)")
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.min_tokens = min_tokens

    def chunk(self, title: str, body: str) -> list[TextChunk]:
        sections = self._sections(body)
        if not sections:
            sections = [_Section(heading=None, text=title)] if title else []
        chunks: list[TextChunk] = []
        for section in sections:
            for piece in self._pack(self._units(section.text)):
                header = " > ".join(p for p in (title, section.heading) if p)
                text = f"{header}\n{piece}" if header and piece != header else piece
                chunks.append(TextChunk(len(chunks), text, section.heading, count_tokens(text)))
        return self._merge_small(chunks)

    def _sections(self, body: str) -> list[_Section]:
        body = body.strip()
        if not body:
            return []
        matches = list(_HEADING_RE.finditer(body))
        if not matches:
            return [_Section(None, body)]
        sections: list[_Section] = []
        if matches[0].start() > 0:
            lead = body[: matches[0].start()].strip()
            if lead:
                sections.append(_Section(None, lead))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            text = body[match.end() : end].strip()
            if text:
                sections.append(_Section(match.group(2).strip(), text))
        return sections

    def _units(self, text: str) -> list[str]:
        """Paragraphs, with oversized paragraphs broken into sentences, then into word windows."""
        units: list[str] = []
        for para in (p.strip() for p in text.split("\n\n")):
            if not para:
                continue
            if count_tokens(para) <= self.target:
                units.append(para)
                continue
            for sentence in split_sentences(para):
                if count_tokens(sentence) <= self.target:
                    units.append(sentence)
                else:
                    units.extend(self._word_windows(sentence))
        return units

    def _word_windows(self, text: str) -> list[str]:
        words = text.split()
        size = max(1, int(self.target / 1.33))
        step = max(1, size - int(self.overlap / 1.33))
        return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]

    def _pack(self, units: list[str]) -> list[str]:
        out: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            tokens = count_tokens(unit)
            if current and current_tokens + tokens > self.target:
                out.append("\n\n".join(current))
                current, current_tokens = self._overlap_tail(current)
            current.append(unit)
            current_tokens += tokens
        if current:
            out.append("\n\n".join(current))
        return out

    def _overlap_tail(self, units: list[str]) -> tuple[list[str], int]:
        if self.overlap == 0:
            return [], 0
        tail: list[str] = []
        tokens = 0
        for unit in reversed(units):
            t = count_tokens(unit)
            if tokens + t > self.overlap:
                break
            tail.insert(0, unit)
            tokens += t
        return tail, tokens

    def _merge_small(self, chunks: list[TextChunk]) -> list[TextChunk]:
        if len(chunks) < 2:
            return chunks
        merged: list[TextChunk] = []
        for chunk in chunks:
            if (
                merged
                and chunk.token_count < self.min_tokens
                and merged[-1].token_count + chunk.token_count <= self.target + self.overlap
                and merged[-1].heading == chunk.heading
            ):
                prev = merged[-1]
                text = prev.text + "\n\n" + chunk.text.split("\n", 1)[-1]
                merged[-1] = TextChunk(prev.ordinal, text, prev.heading, count_tokens(text))
            else:
                merged.append(TextChunk(len(merged), chunk.text, chunk.heading, chunk.token_count))
        return merged
