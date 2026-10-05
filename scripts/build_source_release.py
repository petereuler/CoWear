#!/usr/bin/env python3
"""Stage only the audited CoWear source tree for a GitHub repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ALLOWLIST = (
    "README.md", ".gitignore", "LICENSE", "NOTICE", "CITATION.cff", "pyproject.toml",
    "configs", "src", "scripts", "tests", "splits/paper_v1_session_split.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    for relative in ALLOWLIST:
        source = args.source_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            shutil.copy2(source, destination)
    print(f"staged {len(ALLOWLIST)} public paths under {args.output_root}")


if __name__ == "__main__":
    main()
