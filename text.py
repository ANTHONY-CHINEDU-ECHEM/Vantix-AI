"""Text normalisation, tokenisation and token counting utilities."""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[\w][\w'.+#]*[\w+#]|[\w]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if
    in into is it its just me my not of on or our so than that the their them then there
    these they this to too us was we were what when where which who why will with would you
    your about after all also any because before being both each few more most other over
    same should some such through under until very via while""".split()
)

_SUFFIXES = (
    "ization",
    "ational",
    "fulness",
    "ousness",
    "iveness",
    "ations",
    "ation",
    "ments",
    "ment",
    "ness",
    "ings",
    "ing",
    "ies",
    "ied",
    "ers",
    "er",
    "ed",
    "es",
    "ly",
    "s",
)


def normalise(text: str) -> str:
    """Strip markup, decode entities, apply NFKC and collapse whitespace while keeping paragraphs."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(_WS_RE.sub(" ", line).strip() for line in text.splitlines())
    return _NL_RE.sub("\n\n", text).strip()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """A small, deterministic suffix stripper. Good enough for BM25 recall, cheap at scale.

    Inflected and base forms converge: take and takes, release and released, run and running.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    root = token
    if len(token) > 4:
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                root = token[: -len(suffix)]
                if suffix == "ies":
                    root += "y"
                elif suffix == "es" and not root.endswith(("s", "x", "z", "ch", "sh")):
                    root = token[:-1]
                break
    if len(root) > 2 and root[-1] == root[-2] and root[-1] not in "lsz" and root != token:
        root = root[:-1]
    if len(root) > 4 and root.endswith("e"):
        root = root[:-1]
    return root


def tokenize(text: str, *, remove_stopwords: bool = True, stemming: bool = True) -> list[str]:
    tokens = [t.lower().strip(".'") for t in _WORD_RE.findall(text)]
    out: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok) if stemming else tok)
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class TokenCounter:
    """Counts tokens with tiktoken when installed, otherwise with a calibrated estimate."""

    def __init__(self) -> None:
        self._encoder = None
        try:  # pragma: no cover depends on optional dependency
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:  # pragma: no cover
            return len(self._encoder.encode(text, disallowed_special=()))
        words = len(text.split())
        return max(1, int(round(words * 1.33)))


_COUNTER: TokenCounter | None = None


def count_tokens(text: str) -> int:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = TokenCounter()
    return _COUNTER.count(text)
