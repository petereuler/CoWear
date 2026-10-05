"""Configuration and protocol validation for the public CoWear release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .. import (
    DATASET_VERSION,
    INPUT_FEATURES,
    MODEL_HIDDEN_SIZE,
    PAPER_ROLES,
    PROTOCOL_VERSION,
    TRAIN_BATCH_SIZE,
    TRAIN_MAX_EPOCHS,
)


@dataclass(frozen=True)
class PaperConfig:
    dataset_version: str = DATASET_VERSION
    protocol_version: str = PROTOCOL_VERSION
    sample_rate_hz: float = 100.0
    seed: int = 2027
    target_role: str = "phone"
    roles: tuple[str, ...] = PAPER_ROLES
    split_counts: tuple[int, int, int] = (217, 31, 115)
    input_features: int = INPUT_FEATURES
    hidden_size: int = MODEL_HIDDEN_SIZE
    batch_size: int = TRAIN_BATCH_SIZE
    max_epochs: int = TRAIN_MAX_EPOCHS
    optimizer: str = "AdamW"
    event_segmentation: str = "joint_vertical_gyro_consensus"
    metric: str = "median per-session horizontal ATE"

    def validate(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise ValueError(f"unsupported dataset version: {self.dataset_version}")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.target_role not in self.roles:
            raise ValueError("target_role must be one of roles")
        if sum(self.split_counts) != 363:
            raise ValueError("paper protocol requires 363 sessions")
        if self.input_features != INPUT_FEATURES:
            raise ValueError("paper protocol requires six-channel device-frame input")
        if self.hidden_size != MODEL_HIDDEN_SIZE or self.batch_size != TRAIN_BATCH_SIZE:
            raise ValueError("model size and batch size must match the paper protocol")
        if self.max_epochs != TRAIN_MAX_EPOCHS or self.optimizer != "AdamW":
            raise ValueError("training defaults must match the paper protocol")
        if self.event_segmentation != "joint_vertical_gyro_consensus":
            raise ValueError("unsupported event segmentation protocol")
        if self.metric != "median per-session horizontal ATE":
            raise ValueError("unsupported paper metric")


def load_config(path: Path | None) -> PaperConfig:
    """Load a deliberately small TOML-like JSON config without extra deps."""
    if path is None:
        config = PaperConfig()
        config.validate()
        return config
    values: dict[str, object] = {}
    text = path.read_text(encoding="utf-8")
    # The shipped config is valid JSON as well as TOML-compatible key/value text.
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            values = decoded
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            value = value.strip('"')
            if value.replace(".", "", 1).isdigit():
                values[key] = float(value) if "." in value else int(value)
            else:
                values[key] = value
    config = PaperConfig(
        dataset_version=str(values.get("dataset_version", DATASET_VERSION)),
        protocol_version=str(values.get("protocol_version", PROTOCOL_VERSION)),
        sample_rate_hz=float(values.get("sample_rate_hz", 100.0)),
        seed=int(values.get("seed", 2027)),
        target_role=str(values.get("target_role", "phone")),
        input_features=int(values.get("input_features", INPUT_FEATURES)),
        hidden_size=int(values.get("hidden_size", MODEL_HIDDEN_SIZE)),
        batch_size=int(values.get("batch_size", TRAIN_BATCH_SIZE)),
        max_epochs=int(values.get("max_epochs", TRAIN_MAX_EPOCHS)),
        optimizer=str(values.get("optimizer", "AdamW")),
        event_segmentation=str(values.get("event_segmentation", "joint_vertical_gyro_consensus")),
        metric=str(values.get("metric", "median per-session horizontal ATE")),
    )
    config.validate()
    return config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
