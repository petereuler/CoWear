"""Frozen dataset, geometry, metric, and checkpoint contracts."""

from .config import PaperConfig, load_config
from .data import paper_role, read_manifest, storage_role, validate_paper_split
from .geometry import horizontal_ate, information_fusion, local_to_world_displacement, propagate_gyro, world_to_local_displacement
from .checkpoints import CheckpointMetadata, read_checkpoint

__all__ = [
    "PaperConfig", "load_config", "paper_role", "read_manifest", "storage_role",
    "validate_paper_split", "horizontal_ate", "information_fusion",
    "local_to_world_displacement", "propagate_gyro", "world_to_local_displacement",
    "CheckpointMetadata", "read_checkpoint",
]
