#!/usr/bin/env python3
"""Create a relative-path, SHA-256 manifest for GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cowear import DATASET_URL, DATASET_VERSION, PROTOCOL_VERSION, __version__


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.weights_root.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".xml", ".json"}:
            continue
        files.append({"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size})
    payload = {
        "package_version": __version__,
        "dataset_url": DATASET_URL,
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
