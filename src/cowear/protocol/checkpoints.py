"""Portable checkpoint metadata and release-manifest validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .. import DATASET_VERSION, INPUT_FEATURES, PROTOCOL_VERSION


@dataclass(frozen=True)
class CheckpointMetadata:
    model_version: str
    role: str
    target: str
    feature_frame: str
    split: str
    seed: int
    input_features: int = INPUT_FEATURES
    dataset_version: str = DATASET_VERSION
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise ValueError("checkpoint dataset version does not match public release")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("checkpoint protocol version does not match public release")
        if self.input_features != INPUT_FEATURES:
            raise ValueError("checkpoint must use the paper's six-channel device input")
        if self.feature_frame != "device":
            raise ValueError("checkpoint must declare feature_frame='device'")
        if not self.model_version or not self.role or not self.target:
            raise ValueError("checkpoint metadata is incomplete")


def read_checkpoint(path: Path, *, expected: CheckpointMetadata | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    raw = payload.get("metadata", payload)
    required = ("model_version", "role", "target", "feature_frame", "split", "seed", "input_features")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"checkpoint {path} is missing metadata: {', '.join(missing)}")
    metadata = CheckpointMetadata(
        model_version=str(raw["model_version"]),
        role=str(raw["role"]),
        target=str(raw["target"]),
        feature_frame=str(raw["feature_frame"]),
        split=str(raw["split"]),
        seed=int(raw["seed"]),
        input_features=int(raw["input_features"]),
        dataset_version=str(raw.get("dataset_version", DATASET_VERSION)),
        protocol_version=str(raw.get("protocol_version", PROTOCOL_VERSION)),
    )
    metadata.validate()
    if expected is not None and metadata != expected:
        raise ValueError(f"checkpoint metadata mismatch: {path}")
    payload["metadata"] = metadata
    return payload
