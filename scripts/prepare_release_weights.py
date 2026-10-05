#!/usr/bin/env python3
"""Copy existing paper checkpoints into a release tree with explicit metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cowear import DATASET_VERSION, INPUT_FEATURES, PAPER_ROLE_FOR_STORAGE, PROTOCOL_VERSION


def annotate(source: Path, destination: Path, *, role: str, target: str, feature_frame: str) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint: {source}")
    payload["metadata"] = {
        "model_version": str(payload.get("model_version", "paper_cowear_lstm_checkpoint_v1")),
        "role": role,
        "target": target,
        "feature_frame": feature_frame,
        "split": "train_val_test_session_split",
        "seed": 2027,
        "input_features": INPUT_FEATURES,
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for role in ("mobile", "watch", "rokid"):
        source = args.source_root / "task1_reference_self_checkpoints" / f"{role}.pt"
        public_role = PAPER_ROLE_FOR_STORAGE[role]
        annotate(source, args.output_root / "task1_cowear_lstm_self_checkpoints" / f"{role}.pt", role=public_role, target=public_role, feature_frame="device")
    for target, directory in (("mobile", "task2_phone_target_checkpoints"), ("watch", "task2_watch_target_checkpoints"), ("rokid", "task2_glasses_target_checkpoints")):
        for role in ("mobile", "rokid", "watch_same", "watch_different"):
            source = args.source_root / directory / f"{role}.pt"
            storage_observer = role.removesuffix("_same").removesuffix("_different")
            public_observer = PAPER_ROLE_FOR_STORAGE[storage_observer]
            public_target = PAPER_ROLE_FOR_STORAGE[target]
            annotate(source, args.output_root / directory / f"{role}.pt", role=public_observer, target=public_target, feature_frame="device")
    print(f"prepared release weights under {args.output_root}")


if __name__ == "__main__":
    main()
