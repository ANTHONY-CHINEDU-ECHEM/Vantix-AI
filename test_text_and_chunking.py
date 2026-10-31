from vantix.ingestion.chunker import Chunker
from vantix.text import normalise, stem, tokenize


def test_normalise_strips_markup_and_keeps_paragraphs() -> None:
    text = "<p>Hello&nbsp;<b>world</b></p>\n\n\n\n<script>alert(1)</script>Next"
    out = normalise(text)
    assert "<" not in out and "alert" not in out
    assert "Hello world" in out


def test_tokenize_removes_stopwords_and_stems() -> None:
    assert tokenize("The deployments are running") == ["deploy", "run"]
    assert stem("takes") == stem("take")
    assert stem("released") == stem("release")
    assert stem("policies") == "policy"
    assert "the" not in tokenize("the release")


def test_chunker_respects_headings_and_budget() -> None:
    body = (
        "# Intro\n"
        + " ".join(["word"] * 50)
        + "\n\n# Details\n"
        + "\n\n".join(" ".join(["detail"] * 60) for _ in range(6))
    )
    chunks = Chunker(target_tokens=120, overlap_tokens=20, min_tokens=10).chunk("Guide", body)
    assert len(chunks) > 2
    assert chunks[0].heading == "Intro"
    assert all(c.text.startswith("Guide >") for c in chunks)
    assert all(c.token_count <= 120 + 40 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunker_handles_title_only() -> None:
    chunks = Chunker().chunk("Only a title", "")
    assert len(chunks) == 1 and chunks[0].text == "Only a title"


def test_chunker_splits_giant_paragraph() -> None:
    chunks = Chunker(target_tokens=100, overlap_tokens=10).chunk("T", " ".join(["x"] * 2000))
    assert len(chunks) >= 15
