#!/usr/bin/env python3
"""Verify that a release tag, pyproject.toml, the package and the Helm chart agree on the version.

Usage:
    python scripts/check_version.py v1.2.3
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^v?(\d+\.\d+\.\d+(?:rc\d+|a\d+|b\d+)?)$")


def main() -> int:
    if len(sys.argv) != 2 or not SEMVER.match(sys.argv[1]):
        print("usage: check_version.py vMAJOR.MINOR.PATCH")
        return 2
    tag = SEMVER.match(sys.argv[1]).group(1)  # type: ignore[union-attr]
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    init = re.search(r'__version__ = "([^"]+)"', (ROOT / "src/vantix/__init__.py").read_text()).group(1)  # type: ignore[union-attr]
    chart_text = (ROOT / "deploy/helm/vantix/Chart.yaml").read_text()
    chart = re.search(r"^version:\s*(\S+)", chart_text, re.M).group(1)  # type: ignore[union-attr]
    app = re.search(r'^appVersion:\s*"?([^"\s]+)"?', chart_text, re.M).group(1)  # type: ignore[union-attr]
    versions = {"tag": tag, "pyproject": pyproject, "package": init, "chart": chart, "chart appVersion": app}
    if len(set(versions.values())) != 1:
        print("Version mismatch:")
        for name, value in versions.items():
            print(f"  {name}: {value}")
        return 1
    print(f"All versions agree on {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
