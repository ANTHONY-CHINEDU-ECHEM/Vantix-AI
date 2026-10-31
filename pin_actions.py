#!/usr/bin/env python3
"""Pin every GitHub Action in .github to an immutable commit SHA.

Tags such as ``@v4`` can be moved by whoever controls the upstream repository, which is a
supply chain risk. This script resolves each reference to the newest release tag within the
same major version, replaces it with the full commit SHA and records the version in a
trailing comment (``uses: actions/checkout@<sha> # v4.2.2``). Dependabot understands this
format and keeps the pins current.

Usage:
    python scripts/pin_actions.py          # rewrite files in place
    python scripts/pin_actions.py check    # exit 1 if any action is not pinned to a SHA
"""

from __future__ import annotations

import re
import subprocess  # nosec B404: runs git with fixed arguments only
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USES = re.compile(
    r"(?P<prefix>uses:\s*)(?P<repo>[\w.\-]+/[\w.\-]+)(?P<path>/[\w./\-]+)?@(?P<ref>[\w.\-]+)(?P<comment>\s*#.*)?$"
)
SHA = re.compile(r"^[0-9a-f]{40}$")


def version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", tag))


@cache
def resolve(repo: str, ref: str) -> tuple[str, str]:
    """Return (sha, tag) for the newest tag matching the major version of ``ref``."""
    output = subprocess.run(  # nosec B603 B607: fixed git invocation
        ["git", "ls-remote", "--tags", f"https://github.com/{repo}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    peeled: dict[str, str] = {}
    direct: dict[str, str] = {}
    for line in output.splitlines():
        sha, name = line.split("\t")
        tag = name.removeprefix("refs/tags/")
        if tag.endswith("^{}"):
            peeled[tag[:-3]] = sha
        else:
            direct[tag] = sha
    commits = {tag: peeled.get(tag, sha) for tag, sha in direct.items()}
    major = re.match(r"v?(\d+)", ref)
    if ref in commits and not re.fullmatch(r"v?\d+", ref):
        return commits[ref], ref
    if major is None:
        raise SystemExit(f"cannot interpret ref {repo}@{ref}")
    prefix = "v" if ref.startswith("v") else ""
    candidates = [t for t in commits if re.fullmatch(rf"{prefix}{major.group(1)}(\.\d+)*", t)]
    if not candidates:
        raise SystemExit(f"no release tags found for {repo}@{ref}")
    best = max(candidates, key=version_key)
    return commits[best], best


def process(check_only: bool) -> int:
    unpinned = 0
    files = sorted((ROOT / ".github").rglob("*.yml")) + sorted((ROOT / ".github").rglob("*.yaml"))
    for path in files:
        lines = path.read_text().splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            match = USES.search(line.rstrip("\n"))
            if not match or match.group("repo").startswith("./") or line.strip().startswith("uses: ./"):
                continue
            if "docker://" in line:
                continue
            ref = match.group("ref")
            if SHA.match(ref):
                continue
            unpinned += 1
            if check_only:
                print(f"{path.relative_to(ROOT)}:{i + 1}: {match.group('repo')}@{ref} is not pinned to a SHA")
                continue
            sha, tag = resolve(match.group("repo"), ref)
            new = f"{match.group('prefix')}{match.group('repo')}{match.group('path') or ''}@{sha} # {tag}"
            lines[i] = line[: match.start()] + new + "\n"
            changed = True
            print(f"{path.relative_to(ROOT)}: {match.group('repo')}@{ref} -> {sha[:12]} ({tag})")
        if changed:
            path.write_text("".join(lines))
    if check_only and unpinned:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(process(check_only=len(sys.argv) > 1 and sys.argv[1] == "check"))
