#!/usr/bin/env python3
"""Enforce the README house style in CI.

Rules (all mandatory):
1. No hyphens, en dashes, em dashes, minus signs or other dash characters anywhere.
2. No table of contents (no "contents" heading and no lists of in page anchor links).
3. Mermaid architecture diagrams carry no colour and no background:
   * no colour values (hex, rgb, hsl or named colours) and no init theme overrides
   * every diagram declares ``classDef default fill:none``
   * every subgraph is styled ``fill:none``
   * no edge labels (Mermaid paints a background box behind them)
   * no fill other than ``none`` anywhere
4. Markdown tables are not used, because their delimiter rows require hyphens.

Usage:
    python scripts/check_readme.py [PATH]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DASHES = {
    "-": "hyphen",
    "‐": "hyphen",
    "‑": "non breaking hyphen",
    "‒": "figure dash",
    "–": "en dash",
    "—": "em dash",
    "―": "horizontal bar",
    "−": "minus sign",
    "﹘": "small em dash",
    "﹣": "small hyphen",
    "－": "fullwidth hyphen",
}
COLOUR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\b(?:color|stroke|background)\s*:", re.I)
FILL_RE = re.compile(r"fill\s*:\s*([^,;\s]+)", re.I)


def check(text: str) -> list[str]:
    problems: list[str] = []
    lines = text.splitlines()

    for number, line in enumerate(lines, 1):
        for char, name in DASHES.items():
            if char in line:
                problems.append(f"line {number}: contains a {name}: {line.strip()[:100]}")

    for number, line in enumerate(lines, 1):
        if re.match(r"^#{1,6}\s+(table of )?contents\b", line.strip(), re.I):
            problems.append(f"line {number}: table of contents heading is not allowed")
    anchor_links = [n for n, line in enumerate(lines, 1) if re.search(r"\]\(#[^)]+\)", line)]
    if len(anchor_links) >= 3:
        problems.append(f"lines {anchor_links[:5]}: in page anchor lists look like a table of contents")

    blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    if not blocks:
        problems.append("no Mermaid architecture diagram found")
    for index, block in enumerate(blocks, 1):
        label = f"mermaid diagram {index}"
        if "%%{init" in block:
            problems.append(f"{label}: init directives (theme overrides) are not allowed")
        if COLOUR_RE.search(block):
            problems.append(f"{label}: colour values are not allowed: {COLOUR_RE.search(block).group(0)}")  # type: ignore[union-attr]
        for fill in FILL_RE.findall(block):
            if fill.lower() != "none":
                problems.append(f"{label}: fill must be none, found fill:{fill}")
        if not re.search(r"^\s*classDef default fill:none\s*$", block, re.M):
            problems.append(f"{label}: missing 'classDef default fill:none'")
        for subgraph in re.findall(r"^\s*subgraph\s+(\w+)", block, re.M):
            if not re.search(rf"^\s*style {subgraph} fill:none\s*$", block, re.M):
                problems.append(f"{label}: subgraph {subgraph} is not styled fill:none")
        if re.search(r"(==>|-->|-\.->)\|", block) or re.search(r"\|\s*\"", block):
            problems.append(f"{label}: edge labels are not allowed (they render with a background box)")

    if re.search(r"^\s*\|?\s*:?-{3,}:?\s*\|", text, re.M):
        problems.append("markdown tables are not allowed; use HTML tables")
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "README.md"
    problems = check(path.read_text(encoding="utf-8"))
    if problems:
        print(f"README policy check failed for {path} with {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"README policy check passed for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
