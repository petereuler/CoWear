#!/usr/bin/env python3
"""Train the official RoNIN ResNet-18 architecture on the CoWear split.

The network implementation is imported directly from Navae/RONIN/source.  This
adapter only converts CoWear's aligned IMU and Vicon streams into RoNIN's 6D
world-frame gyro/accelerometer input and planar velocity supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from ..protocol import time as ricloc_protocol

REPO_ROOT = SCRIPT_DIR.parents[2]
from ..models.ronin_resnet import BasicBlock1D, FCOutputModule, ResNet1D
from ..baselines import _pdr_support as pdr_adapter

# Attribution/provenance string for the self-contained RoNIN architecture.
OFFICIAL_SOURCE = "https://github.com/nesl/RoNIN"


DATA_LOADER_VERSION = "cowear_manifest_align_v3_ricloc_clock_device_heading"
VALID_SPLITS = {"train", "val", "test"}
NAV_FROM_VICON_QUAT_XYZW = np.asarray(
    [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64
)
@dataclass
class Sequence:
    base_id: str
    split: str
    same_hand: str
    features: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    truth_valid: np.ndarray
    dt_s: float
    target_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    target_std: np.ndarray = field(
        default_factory=lambda: np.ones(2, dtype=np.float32)
    )


def read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def rows_by_base_id(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed = {}
    for row in rows:
        base_id = row.get("base_id", "").strip()
        if not base_id:
            raise ValueError(f"Missing base_id in {path}")
        if base_id in indexed:
            raise ValueError(f"Duplicate base_id={base_id} in {path}")
        indexed[base_id] = row
    return indexed


def manifest_bool(row: dict[str, str], field: str, base_id: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Invalid {field}={value!r} for {base_id}")
    return value == "true"


def available_roles_from_row(row: dict[str, str], base_id: str) -> tuple[str, ...]:
    """Read role availability from the published session manifest."""
    listed = tuple(part.strip() for part in row.get("available_roles", "").split(",") if part.strip())
    if listed:
        unknown = set(listed) - set(pdr_adapter.ROLES)
        if unknown:
            raise ValueError(f"Unknown available_roles={sorted(unknown)} for {base_id}")
        return listed
    roles = tuple(role for role in pdr_adapter.ROLES if manifest_bool(row, f"{role}_available", base_id))
    return roles


def read_manifest_specs(manifest: Path, split_index: Path):
    """Join authoritative manifest metadata with split labels by base_id."""
    manifest_rows = read_csv_rows(manifest, "manual aligned dataset manifest")
    split_rows = read_csv_rows(split_index, "session split index")
    manifest_by_id = rows_by_base_id(manifest_rows, manifest)
    split_by_id = rows_by_base_id(split_rows, split_index)

    missing_split = sorted(set(manifest_by_id) - set(split_by_id))
    unknown_split = sorted(set(split_by_id) - set(manifest_by_id))
    if missing_split or unknown_split:
        raise ValueError(
            "Manifest/split base_id mismatch: "
            f"missing_split={missing_split[:5]} unknown_split={unknown_split[:5]}"
        )

    specs = []
    for row in manifest_rows:
        base_id = row["base_id"].strip()
        split = split_by_id[base_id].get("split", "").strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split={split!r} for {base_id} in {split_index}")
        available_roles = available_roles_from_row(row, base_id)
        # Legacy manifests include explicit availability and missing-role
        # columns; older manifests encode the same information in
        # available_roles and per-device file columns.
        if any(f"{role}_available" in row for role in pdr_adapter.ROLES):
            missing_roles = tuple(part for part in row.get("missing_roles", "").split(",") if part)
            for role in pdr_adapter.ROLES:
                available_flag = manifest_bool(row, f"{role}_available", base_id)
                if (role in available_roles) != available_flag:
                    raise ValueError(f"Manifest available_roles conflicts with {role}_available for {base_id}")
                if (role in missing_roles) == available_flag:
                    raise ValueError(f"Manifest missing_roles conflicts with {role}_available for {base_id}")
        specs.append(
            pdr_adapter.SessionSpec(
                base_id=base_id,
                date=row["date"],
                session_id=row.get("session_id") or base_id.rsplit("/", 1)[-1],
                split=split,
                available_roles=available_roles,
                watch_primary_file=(
                    row.get("watch_primary_file", "")
                    or row.get("watch_file", "")
                    or Path(row.get("watch_measure_file", "")).name
                    or "imu.jsonl"
                ),
                same_hand=row.get("same_hand", ""),
                sample_quality=row.get("sample_quality", ""),
            )
        )
    return specs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_provenance(args) -> dict[str, str]:
    cached = getattr(args, "_data_provenance", None)
    if cached is not None:
        return cached
    truth_time_alignment = getattr(args, "truth_time_alignment", "manifest")
    provenance = {
        "data_loader_version": DATA_LOADER_VERSION,
        "processed_root": str(args.processed_root.resolve()),
        "manifest_source": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "split_source": str(args.split_index.resolve()),
        "split_sha256": file_sha256(args.split_index),
        "time_protocol": ricloc_protocol.TIME_PROTOCOL_VERSION,
        "imu_time_field": "alignedRelativeS + alignment_info.parameters.windowStartMs/1000",
        "truth_time_field": "timestamp_ms/1000",
        "stored_time_origin": "alignment_info.parameters.windowStartMs/1000",
        "truth_time_alignment": truth_time_alignment,
    }
    if truth_time_alignment == "mobile_vicon_lag":
        lag_file = getattr(args, "mobile_vicon_lag_file", None)
        if lag_file is None or not lag_file.is_file():
            raise FileNotFoundError(f"Missing mobile Vicon/IMU lag audit: {lag_file}")
        provenance["mobile_vicon_lag_file"] = str(lag_file.resolve())
        provenance["mobile_vicon_lag_sha256"] = file_sha256(lag_file)
    args._data_provenance = provenance
    return provenance


def aligned_sources(processed_root: Path, spec, role: str) -> dict[str, Path]:
    session_dir = processed_root / spec.base_id
    imu_name = spec.watch_primary_file if role == "watch" else "imu.jsonl"
    return {
        "session_dir": session_dir,
        "imu_source": session_dir / "measure" / "align" / role / imu_name,
        "truth_source": session_dir / "groundtruth" / "align.csv",
        "alignment_info_source": session_dir / "alignment_info.json",
    }


def validate_aligned_session(processed_root: Path, spec, role: str) -> dict[str, Path]:
    sources = aligned_sources(processed_root, spec, role)
    for field in ("imu_source", "truth_source", "alignment_info_source"):
        if not sources[field].is_file():
            raise FileNotFoundError(f"Manifest marks {role} available but {field} is missing: {sources[field]}")
    try:
        alignment_info = json.loads(sources["alignment_info_source"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid alignment_info.json: {sources['alignment_info_source']}") from error
    availability = alignment_info.get("deviceAvailability")
    if isinstance(availability, dict) and role not in availability.get("availableRoles", []):
        raise ValueError(
            f"Manifest marks {role} available but alignment_info does not: {spec.base_id}"
        )
    outputs = alignment_info.get("outputs")
    if isinstance(outputs, dict) and outputs.get("groundtruthAlign") != "groundtruth/align.csv":
        raise ValueError(f"Unexpected aligned truth output for {spec.base_id}")
    if isinstance(outputs, dict) and outputs.get("measureAlign") != "measure/align":
        raise ValueError(f"Unexpected aligned measurement output for {spec.base_id}")
    if "manualAlignmentMaterialization" in alignment_info and not isinstance(
        alignment_info.get("manualAlignmentMaterialization"), dict
    ):
        raise ValueError(f"Missing manual alignment materialization for {spec.base_id}")
    return sources


def cache_provenance(args, spec) -> dict[str, str]:
    sources = aligned_sources(args.processed_root, spec, args.role)
    return {
        **data_provenance(args),
        "base_id": spec.base_id,
        "imu_source": str(sources["imu_source"].resolve()),
        "truth_source": str(sources["truth_source"].resolve()),
        "alignment_info_source": str(sources["alignment_info_source"].resolve()),
        "truth_time_offset_s": f"{truth_time_offset(args, spec.base_id, args.role):.6f}",
    }


def mobile_vicon_lag_offsets(args) -> dict[str, float]:
    cached = getattr(args, "_mobile_vicon_lag_offsets", None)
    if cached is not None:
        return cached
    if not args.mobile_vicon_lag_file.is_file():
        raise FileNotFoundError(f"Missing mobile Vicon/IMU lag audit: {args.mobile_vicon_lag_file}")
    offsets = {}
    for row in read_csv_rows(args.mobile_vicon_lag_file, "mobile Vicon/IMU lag audit"):
        try:
            offsets[row["base_id"]] = float(row["best_lag_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid lag row in {args.mobile_vicon_lag_file}: {row}") from error
    args._mobile_vicon_lag_offsets = offsets
    return offsets


def truth_time_offset(args, base_id: str, role: str) -> float:
    """Return lag where Vicon(t + lag) is synchronized to role IMU(t)."""
    if args.truth_time_alignment == "manifest":
        return 0.0
    if args.truth_time_alignment != "mobile_vicon_lag":
        raise ValueError(f"Unsupported truth-time alignment: {args.truth_time_alignment}")
    if role != "mobile":
        return 0.0
    offsets = mobile_vicon_lag_offsets(args)
    if base_id not in offsets:
        raise ValueError(f"Mobile Vicon/IMU lag audit lacks {base_id}")
    return offsets[base_id]


class RoNINWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[Sequence],
        window_size: int,
        stride: int,
        augment: bool,
        supervision_mode: str,
    ):
        self.sequences = sequences
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.supervision_mode = supervision_mode
        self.index = []
        for sequence_id, sequence in enumerate(sequences):
            if supervision_mode == "ronin_window":
                indices = valid_window_starts(sequence, window_size, stride)
            elif supervision_mode == "instantaneous":
                indices = valid_frame_ids(sequence, window_size, stride)
            else:
                raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
            for index in indices:
                self.index.append((sequence_id, int(index)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        sequence_id, frame_or_start = self.index[item]
        sequence = self.sequences[sequence_id]
        if self.supervision_mode == "ronin_window":
            start = frame_or_start
            feature = sequence.features[start:start + self.window_size].copy()
            target = (
                (sequence.position[start + self.window_size] - sequence.position[start])
                / (sequence.dt_s * self.window_size)
            ).astype(np.float32, copy=False)
        else:
            frame = frame_or_start
            feature = sequence.features[frame - self.window_size:frame].copy()
            target = sequence.velocity[frame].copy()
        target = (target - sequence.target_mean) / sequence.target_std
        if self.augment:
            angle = random.uniform(-math.pi, math.pi)
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
            feature[:, (0, 1)] = feature[:, (0, 1)] @ rotation.T
            feature[:, (3, 4)] = feature[:, (3, 4)] @ rotation.T
            target = target @ rotation.T
        return torch.from_numpy(feature.T), torch.from_numpy(target)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--role", choices=("mobile", "watch", "rokid"), default="mobile")
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Authoritative manual aligned dataset manifest (defaults inside --processed-root).",
    )
    parser.add_argument("--split-index", type=Path, default=REPO_ROOT / "data" / "splits" / "session_split_seed2027.csv")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--initial-heading-distance-m",
        type=float,
        default=1.5,
        help="Ground-truth displacement used to define the known initial walking heading.",
    )
    parser.add_argument(
        "--truth-max-gap-s",
        type=float,
        default=0.05,
        help="Maximum gap allowed inside a truth-valid Vicon interval.",
    )
    parser.add_argument(
        "--truth-time-alignment",
        choices=("manifest", "mobile_vicon_lag"),
        default="manifest",
        help=(
            "Truth timestamp policy. mobile_vicon_lag applies the per-session residual "
            "lag measured from mobile gyro and Vicon angular speed."
        ),
    )
    parser.add_argument(
        "--mobile-vicon-lag-file",
        type=Path,
        default=REPO_ROOT / "CoWear" / "diagnostics" / "mobile_vicon_imu_lag.csv",
        help="CSV produced by audit_mobile_vicon_imu_lag.py.",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.2,
        help="Maximum native-stream gap allowed inside a converted sequence.",
    )
    parser.add_argument(
        "--train-gap-policy",
        choices=("interpolate", "longest_contiguous"),
        default="interpolate",
        help="How to handle natural sampling gaps in training sessions.",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--supervision-mode",
        choices=("ronin_window", "instantaneous"),
        default="ronin_window",
        help=(
            "Velocity supervision target. ronin_window matches official RoNIN "
            "ResNet: (pos[t+window]-pos[t]) over the input window duration. "
            "instantaneous keeps the legacy endpoint-gradient label."
        ),
    )
    parser.add_argument(
        "--imu-moving-average-window",
        type=int,
        default=5,
        help="Causal moving-average length applied to resampled accelerometer and gyroscope samples.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        help="Optional official RoNIN checkpoint used to initialize the ResNet-18.",
    )
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--coordinate-frame",
        choices=("initial_device_yaw_local", "initial_motion_heading_local"),
        default="initial_device_yaw_local",
        help=(
            "Canonical horizontal frame. initial_device_yaw_local keeps the IMU "
            "and Vicon target in the same device-referenced frame; the motion-heading "
            "variant is retained only to reproduce the earlier experiment."
        ),
    )
    parser.add_argument(
        "--input-frame",
        choices=("ronin_world", "body", "imu_heading"),
        default="ronin_world",
        help=(
            "IMU feature frame. ronin_world matches the official global-frame "
            "gyro/accelerometer input; body keeps the glasses/body-frame IMU "
            "and avoids relying on Android-world yaw being aligned to Vicon."
        ),
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--disable-horizontal-augmentation",
        action="store_true",
        help="Disable official random yaw augmentation after canonical local-frame preprocessing.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.imu_moving_average_window < 1:
        parser.error("--imu-moving-average-window must be at least 1")
    if args.manifest is None:
        args.manifest = REPO_ROOT / "data" / "manifest.csv"
    if args.output_root is None:
        suffix = args.role if args.role != "rokid" else "rokid_truth_valid"
        args.output_root = REPO_ROOT / "CoWear" / "results" / f"ronin_{suffix}"
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_frame_ids(sequence: Sequence, window_size: int, stride: int):
    invalid_prefix = np.r_[0, np.cumsum(~sequence.truth_valid.astype(bool))]
    frames = []
    for frame in range(window_size, len(sequence.velocity), stride):
        if invalid_prefix[frame + 1] - invalid_prefix[frame - window_size] == 0:
            frames.append(frame)
    return np.asarray(frames, dtype=np.int64)


def valid_window_starts(sequence: Sequence, window_size: int, stride: int):
    invalid_prefix = np.r_[0, np.cumsum(~sequence.truth_valid.astype(bool))]
    starts = []
    for start in range(0, len(sequence.position) - window_size, stride):
        if invalid_prefix[start + window_size + 1] - invalid_prefix[start] == 0:
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def normalize_quaternions(quat: np.ndarray):
    quat = quat.copy().astype(np.float64)
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat


def interpolate_quaternion(t_src: np.ndarray, quat: np.ndarray, t_out: np.ndarray):
    quat = normalize_quaternions(quat)
    values = np.column_stack([np.interp(t_out, t_src, quat[:, axis]) for axis in range(4)])
    return normalize_quaternions(values)


def rotate_by_quaternion(vectors: np.ndarray, quat: np.ndarray):
    xyz, w = quat[:, :3], quat[:, 3:4]
    cross = 2.0 * np.cross(xyz, vectors)
    return vectors + w * cross + np.cross(xyz, cross)


def yaw_from_quaternion(quat: np.ndarray):
    x, y, z, w = quat.T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for normalized xyzw quaternion rows."""
    left, right = np.asarray(left), np.asarray(right)
    lx, ly, lz, lw = left.T
    rx, ry, rz, rw = right.T
    return normalize_quaternions(
        np.column_stack(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            )
        )
    )


def vicon_to_navigation_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert Vicon y-up xyzw attitudes to the x-z horizontal navigation frame."""
    source = normalize_quaternions(np.asarray(quat, dtype=np.float64))
    transform = np.repeat(NAV_FROM_VICON_QUAT_XYZW[None, :], len(source), axis=0)
    return multiply_quaternions(transform, source)


def read_vicon_quaternions(path: Path, role: str):
    times, quaternions = [], []
    session_dir = ricloc_protocol.session_dir_from_truth_path(path)
    window_start_s = ricloc_protocol.window_start_s(session_dir)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device") != role:
                continue
            try:
                times.append(float(row["timestamp_ms"]) / 1000.0 - window_start_s)
                quaternions.append(
                    [
                        float(row["quat_x"]),
                        float(row["quat_y"]),
                        float(row["quat_z"]),
                        float(row["quat_w"]),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
    return pdr_adapter._sorted_unique(times, quaternions, 4)


def initial_device_yaw(truth_path: Path, role: str, query_time: float) -> float | None:
    times, quaternions = read_vicon_quaternions(truth_path, role)
    if len(times) < 2:
        return None
    quat = interpolate_quaternion(times, quaternions, np.asarray([query_time]))
    return float(yaw_from_quaternion(vicon_to_navigation_quaternion(quat))[0])


def rotate_horizontal(values: np.ndarray, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    output = values.copy()
    output[:, 0] = c * values[:, 0] - s * values[:, 1]
    output[:, 1] = s * values[:, 0] + c * values[:, 1]
    return output


def estimate_heading_anchored_orientation(acc: np.ndarray, gyro: np.ndarray, fs: float, gravity_window: int = 55):
    """Estimate body-to-CoWear-world rotations from IMU only."""
    gravity = causal_moving_average(acc.astype(np.float32), gravity_window)
    gravity /= np.clip(np.linalg.norm(gravity, axis=1, keepdims=True), 1e-6, None)
    rotations = []
    r = Rotation.align_vectors(np.asarray([[0.0, 1.0, 0.0]]), gravity[:1])[0]
    for i in range(len(acc)):
        if i:
            r = r * Rotation.from_rotvec(gyro[i - 1].astype(np.float64) / fs)
        predicted = r.apply(gravity[i])
        correction = Rotation.align_vectors(np.asarray([[0.0, 1.0, 0.0]]), predicted[None])[0]
        r = correction * r
        rotations.append(r)
    initial_x = rotations[0].apply(np.asarray([1.0, 0.0, 0.0]))
    yaw = math.atan2(float(initial_x[2]), float(initial_x[0]))
    anchor = Rotation.from_rotvec(np.asarray([0.0, 1.0, 0.0]) * (-yaw))
    return Rotation.concatenate([anchor * value for value in rotations])


def to_ronin_nav(values: np.ndarray) -> np.ndarray:
    """Convert CoWear world vectors (x,z horizontal, y up) to RoNIN axes."""
    return np.asarray(values)[:, (0, 2, 1)] * np.asarray([1.0, 1.0, -1.0], dtype=np.asarray(values).dtype)


def causal_moving_average(values: np.ndarray, window_size: int):
    if window_size == 1:
        return values.copy()
    padded = np.pad(values, ((window_size - 1, 0), (0, 0)), mode="edge")
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64))
    )
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def initial_heading(position: np.ndarray, distance_m: float = 1.5):
    origin = position[0]
    for point in position[1:]:
        delta = point - origin
        if np.linalg.norm(delta) >= distance_m:
            return math.atan2(float(delta[1]), float(delta[0]))
    delta = position[-1] - origin
    if np.linalg.norm(delta) < 0.5:
        return None
    return math.atan2(float(delta[1]), float(delta[0]))


def cache_path(output_root: Path, base_id: str, role: str):
    return output_root / "cache" / base_id / f"{role}.npz"


def cache_matches_preprocessing(path: Path, args, spec) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected_provenance = cache_provenance(args, spec)
            return (
                "imu_filter" in payload
                and "input_role" in payload
                and "truth_role" in payload
                and "input_frame" in payload
                and str(payload["imu_filter"]) == "causal_moving_average"
                and int(payload["imu_moving_average_window"]) == args.imu_moving_average_window
                and math.isclose(float(payload["dt_s"]), 1.0 / args.sample_rate_hz)
                and str(payload["role"]) == args.role
                and str(payload["input_role"]) == args.role
                and str(payload["truth_role"]) == args.role
                and str(payload["coordinate_frame"]) == args.coordinate_frame
                and str(payload["input_frame"]) == args.input_frame
                and all(
                    field in payload and str(payload[field]) == value
                    for field, value in expected_provenance.items()
                )
            )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def build_sequence(processed_root: Path, spec, role: str, sample_rate_hz: float, coordinate_frame: str, input_frame: str, max_gap_s: float, train_gap_policy: str, initial_heading_distance_m: float, truth_max_gap_s: float, imu_moving_average_window: int, truth_time_offset_s: float):
    validate_aligned_session(processed_root, spec, role)
    data = pdr_adapter.load_data(processed_root, spec, role, max_gap_s)
    # CoWear's aligned export contains gyro/accelerometer streams but does not
    # provide a device-side orientation quaternion.  Body-frame RoNIN input
    # does not need that optional signal; only the world-frame variant does.
    if data is None or len(data.acc_t) < 2 or len(data.gyro_t) < 2:
        return None
    # The materialized CoWear stream has no game/rotation-vector channel.  For
    # the official world-frame RoNIN protocol, use the aligned Vicon device
    # attitude as an explicit ground-truth orientation fallback; body-frame mode
    # remains available for deployment-style runs.
    if len(data.quat_t) < 2 and input_frame == "ronin_world":
        truth_path = processed_root / spec.base_id / "groundtruth" / "align.csv"
        data.quat_t, data.quat = read_vicon_quaternions(truth_path, role)
        if len(data.quat_t) < 2:
            return None
    if data.role != role:
        raise ValueError(f"Loaded {data.role} data while preparing role={role}")
    truth_t = data.truth_t - truth_time_offset_s
    # Match RIC-Loc: all splits use the longest continuous overlap on the
    # shared aligned clock.  Natural dropouts are never silently bridged by
    # interpolation.
    overlap_streams = [truth_t, data.acc_t, data.gyro_t]
    if input_frame == "ronin_world":
        overlap_streams.append(data.quat_t)
    overlap = ricloc_protocol.longest_multistream_overlap(
        overlap_streams,
        max_gap_s=max_gap_s,
    )
    if overlap is None:
        return None
    start, end = overlap
    if end - start < 10.0:
        return None
    t = np.arange(start, end, 1.0 / sample_rate_hz, dtype=np.float64)
    acc = pdr_adapter.interp_columns(data.acc_t, data.acc, t)
    gyro = pdr_adapter.interp_columns(data.gyro_t, data.gyro, t)
    acc = causal_moving_average(acc, imu_moving_average_window)
    gyro = causal_moving_average(gyro, imu_moving_average_window)
    if len(data.quat_t) >= 2:
        quat = interpolate_quaternion(data.quat_t, data.quat, t)
        world_acc = rotate_by_quaternion(acc, quat)
        inverse_quat = quat.copy()
        inverse_quat[:, :3] *= -1.0
        inverse_acc = rotate_by_quaternion(acc, inverse_quat)
        gravity_score = lambda values: np.sum((np.mean(values, axis=0) - np.asarray([0.0, 0.0, 9.80665])) ** 2)
        if gravity_score(inverse_acc) < gravity_score(world_acc):
            quat = inverse_quat
            world_acc = inverse_acc
        world_gyro = rotate_by_quaternion(gyro, quat)
    else:
        quat = None
        world_acc = acc
        world_gyro = gyro
    position = pdr_adapter.interp_columns(truth_t, data.truth_xy, t)
    truth_intervals = pdr_adapter._continuous_intervals(
        truth_t,
        max_gap_s=truth_max_gap_s,
    )
    truth_valid = np.zeros(len(t), dtype=bool)
    motion_heading = None
    for interval_start, interval_end in truth_intervals:
        selected = (t >= interval_start) & (t <= interval_end)
        if np.count_nonzero(selected) < 3:
            continue
        truth_valid[selected] = True
        if motion_heading is None:
            motion_heading = initial_heading(position[selected], distance_m=initial_heading_distance_m)
    if motion_heading is None:
        return None
    if coordinate_frame == "initial_device_yaw_local":
        target_heading = initial_device_yaw(
            aligned_sources(processed_root, spec, role)["truth_source"], role, float(t[0])
            + truth_time_offset_s,
        )
        if target_heading is None:
            return None
    elif coordinate_frame == "initial_motion_heading_local":
        target_heading = motion_heading
    else:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    if input_frame == "ronin_world":
        # Canonicalize the official global-frame input by removing the
        # device's arbitrary Android-world yaw.
        if quat is None:
            return None
        # CoWear targets are expressed in the initial-motion-heading frame.
        # Rotate the RoNIN navigation features into that same frame; using the
        # device yaw here would introduce an unobservable constant heading
        # mismatch when the device has no orientation sensor.
        initial_imu_yaw = target_heading
        nav_acc = to_ronin_nav(world_acc)
        nav_gyro = to_ronin_nav(world_gyro)
        feature_acc = rotate_horizontal(nav_acc, -initial_imu_yaw)
        feature_gyro = rotate_horizontal(nav_gyro, -initial_imu_yaw)
    elif input_frame == "imu_heading":
        orientation = estimate_heading_anchored_orientation(acc, gyro, sample_rate_hz)
        imu_world_acc = orientation.apply(acc)
        imu_world_gyro = orientation.apply(gyro)
        nav_acc = to_ronin_nav(imu_world_acc)
        nav_gyro = to_ronin_nav(imu_world_gyro)
        feature_acc = rotate_horizontal(nav_acc, 0.0)[:, :]
        feature_gyro = rotate_horizontal(nav_gyro, 0.0)[:, :]
    elif input_frame == "body":
        feature_acc = acc
        feature_gyro = gyro
    else:
        raise ValueError(f"Unsupported input frame: {input_frame}")
    position = position - position[:1]
    position = rotate_horizontal(position, -target_heading)
    velocity = np.zeros_like(position, dtype=np.float32)
    for interval_start, interval_end in truth_intervals:
        selected = np.flatnonzero((t >= interval_start) & (t <= interval_end))
        if len(selected) < 3:
            continue
        velocity[selected] = np.gradient(
            position[selected],
            1.0 / sample_rate_hz,
            axis=0,
        ).astype(np.float32)
    features = np.concatenate((feature_gyro, feature_acc), axis=1).astype(np.float32)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(velocity)):
        return None
    return features, velocity, position.astype(np.float32), truth_valid


def prepare(args, specs):
    provenance = data_provenance(args)
    rows = []
    for index, spec in enumerate(specs, 1):
        if args.role not in spec.available_roles:
            continue
        path = cache_path(args.output_root, spec.base_id, args.role)
        if args.force_prepare or not cache_matches_preprocessing(path, args, spec):
            prepared = build_sequence(
                args.processed_root,
                spec,
                args.role,
                args.sample_rate_hz,
                args.coordinate_frame,
                args.input_frame,
                args.max_gap_s,
                args.train_gap_policy,
                args.initial_heading_distance_m,
                args.truth_max_gap_s,
                args.imu_moving_average_window,
                truth_time_offset(args, spec.base_id, args.role),
            )
            if prepared is None:
                continue
            features, velocity, position, truth_valid = prepared
            cache_metadata = cache_provenance(args, spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                features=features,
                velocity=velocity,
                position=position,
                truth_valid=truth_valid,
                dt_s=1.0 / args.sample_rate_hz,
                role=np.asarray(args.role),
                input_role=np.asarray(args.role),
                truth_role=np.asarray(args.role),
                coordinate_frame=np.asarray(args.coordinate_frame),
                input_frame=np.asarray(args.input_frame),
                imu_filter=np.asarray("causal_moving_average"),
                imu_moving_average_window=np.asarray(args.imu_moving_average_window),
                **{field: np.asarray(value) for field, value in cache_metadata.items()},
            )
        with np.load(path, allow_pickle=False) as payload:
            truth_valid = payload["truth_valid"] if "truth_valid" in payload else np.ones(len(payload["features"]), dtype=bool)
            input_role = str(payload["input_role"]) if "input_role" in payload else str(payload["role"])
            truth_role = str(payload["truth_role"]) if "truth_role" in payload else str(payload["role"])
            input_frame = str(payload["input_frame"]) if "input_frame" in payload else "ronin_world"
            rows.append({"base_id": spec.base_id, "split": spec.split, "same_hand": spec.same_hand, "role": args.role, "input_role": input_role, "truth_role": truth_role, "data_loader_version": str(payload["data_loader_version"]), "manifest_source": str(payload["manifest_source"]), "imu_source": str(payload["imu_source"]), "truth_source": str(payload["truth_source"]), "alignment_info_source": str(payload["alignment_info_source"]), "cache_path": str(path), "samples": str(len(payload["features"])), "valid_samples": str(int(np.count_nonzero(truth_valid))), "coordinate_frame": str(payload["coordinate_frame"]), "input_frame": input_frame})
        if index % 50 == 0 or index == len(specs):
            print(f"[prepare] {index}/{len(specs)}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "cache_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_id", "split", "same_hand", "role", "input_role", "truth_role", "data_loader_version", "manifest_source", "imu_source", "truth_source", "alignment_info_source", "cache_path", "samples", "valid_samples", "coordinate_frame", "input_frame"])
        writer.writeheader()
        writer.writerows(rows)
    # RoNIN assumes standardized inertial channels. Fit statistics on train
    # sessions only to avoid leaking validation/test distributions.
    train_values = []
    for row in rows:
        if row["split"] != "train":
            continue
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            valid = payload["truth_valid"].astype(bool)
            train_values.append(payload["features"][valid])
    if not train_values:
        raise ValueError(f"No training features available for role={args.role}")
    stacked = np.concatenate(train_values, axis=0).astype(np.float64)
    stats = {"mean": stacked.mean(axis=0).tolist(), "std": np.maximum(stacked.std(axis=0), 1e-3).tolist(), "fit_split": "train"}
    (args.output_root / "feature_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    train_targets = []
    for row in rows:
        if row["split"] != "train":
            continue
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            valid = payload["truth_valid"].astype(bool)
            velocity = payload["velocity"]
            if args.supervision_mode == "ronin_window":
                w = int(args.window_size)
                if len(valid) >= w + 1:
                    valid_starts = np.flatnonzero(
                        np.lib.stride_tricks.sliding_window_view(valid, w + 1).all(axis=1)
                    )
                else:
                    valid_starts = np.empty(0, dtype=np.int64)
                if len(valid_starts):
                    position = payload["position"]
                    train_targets.append(
                        (position[valid_starts + w] - position[valid_starts])
                        / ((1.0 / args.sample_rate_hz) * w)
                    )
            else:
                train_targets.append(velocity[valid])
    if not train_targets:
        raise ValueError(f"No valid training targets for supervision_mode={args.supervision_mode}; reduce --window-size or truth gap threshold")
    target_values = np.concatenate(train_targets, axis=0).astype(np.float64)
    target_stats = {"mean": target_values.mean(axis=0).tolist(), "std": np.maximum(target_values.std(axis=0), 1e-3).tolist(), "fit_split": "train"}
    (args.output_root / "target_stats.json").write_text(json.dumps(target_stats, indent=2) + "\n")
    split_counts = {split: sum(row["split"] == split for row in rows) for split in sorted(VALID_SPLITS)}
    (args.output_root / "data_provenance.json").write_text(
        json.dumps(
            {
                **provenance,
                "role": args.role,
                "input_role": args.role,
                "truth_role": args.role,
                "selected_sessions": len(rows),
                "selected_sessions_by_split": split_counts,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return rows


def read_sequences(output_root: Path, split: str, args=None):
    with (output_root / "cache_index.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    sequences = []
    stats_path = output_root / "feature_stats.json"
    stats = None
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        feat_mean = np.asarray(stats["mean"], dtype=np.float32)
        feat_std = np.asarray(stats["std"], dtype=np.float32)
    target_stats_path = output_root / "target_stats.json"
    if target_stats_path.is_file():
        target_stats = json.loads(target_stats_path.read_text())
        target_mean = np.asarray(target_stats["mean"], dtype=np.float32)
        target_std = np.asarray(target_stats["std"], dtype=np.float32)
    else:
        # Legacy/test caches predate target normalization. Keeping identity
        # defaults lets provenance validation report the actionable error.
        target_mean = np.zeros(2, dtype=np.float32)
        target_std = np.ones(2, dtype=np.float32)
    for row in rows:
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            role = str(payload["role"])
            if "input_role" not in payload or "truth_role" not in payload:
                raise ValueError(
                    f"Cache lacks input/truth role provenance: {row['cache_path']}; "
                    "rerun prepare or all"
                )
            input_role = str(payload["input_role"])
            truth_role = str(payload["truth_role"])
            if input_role != role or truth_role != role:
                raise ValueError(
                    f"Cache role mismatch in {row['cache_path']}: "
                    f"role={role} input_role={input_role} truth_role={truth_role}"
                )
            if args is not None:
                expected = {
                    **data_provenance(args),
                    "base_id": row["base_id"],
                    "imu_source": str(
                        (args.processed_root / row["base_id"] / "measure" / "align" / args.role / "imu.jsonl").resolve()
                    ),
                    "truth_source": str(
                        (args.processed_root / row["base_id"] / "groundtruth" / "align.csv").resolve()
                    ),
                    "alignment_info_source": str(
                        (args.processed_root / row["base_id"] / "alignment_info.json").resolve()
                    ),
                }
                mismatches = [
                    field
                    for field, value in expected.items()
                    if field not in payload or str(payload[field]) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"Cache data provenance mismatch in {row['cache_path']}: "
                        f"{', '.join(mismatches)}; rerun prepare or all"
                    )
            truth_valid = payload["truth_valid"] if "truth_valid" in payload else np.ones(len(payload["features"]), dtype=bool)
            features = payload["features"].astype(np.float32)
            if stats is not None:
                features = (features - feat_mean) / feat_std
            sequences.append(Sequence(row["base_id"], row["split"], row["same_hand"], features, payload["velocity"], payload["position"], truth_valid.astype(bool), float(payload["dt_s"]), target_mean, target_std))
    return sequences


def build_model(window_size: int, dropout: float = 0.1):
    return ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64, output_block=FCOutputModule, kernel_size=3, fc_dim=512, in_dim=window_size // 32 + 1, dropout=dropout, trans_planes=128)


def load_pretrained_compatible(model: nn.Module, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(
        key
        for key, value in source_state.items()
        if key not in model_state or model_state[key].shape != value.shape
    )
    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


def evaluate_loss(model, loader, device):
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for feature, target in loader:
            prediction = model(feature.to(device, non_blocking=True))
            loss_sum += float(torch.sum((prediction - target.to(device, non_blocking=True)) ** 2).item())
            count += int(target.numel())
    return loss_sum / max(count, 1)


def train(args, device):
    train_sequences = read_sequences(args.output_root, "train", args)
    val_sequences = read_sequences(args.output_root, "val", args)
    train_set = RoNINWindowDataset(
        train_sequences,
        args.window_size,
        args.stride,
        augment=not args.disable_horizontal_augmentation,
        supervision_mode=args.supervision_mode,
    )
    val_set = RoNINWindowDataset(
        val_sequences,
        args.window_size,
        args.stride,
        augment=False,
        supervision_mode=args.supervision_mode,
    )
    loader_args = {"batch_size": args.batch_size, "num_workers": args.loader_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = build_model(args.window_size, args.dropout).to(device)
    if args.pretrained_model is not None:
        loaded_count, skipped = load_pretrained_compatible(model, args.pretrained_model, device)
        print(
            f"[train] initialized {loaded_count} compatible tensors from official checkpoint "
            f"{args.pretrained_model}; reinitialized {len(skipped)} incompatible tensors",
            flush=True,
        )
        if skipped:
            print(f"[train] incompatible tensors: {', '.join(skipped)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "train_config.json").write_text(
        json.dumps(
            {
                "official_source": str(OFFICIAL_SOURCE),
                **data_provenance(args),
                "role": args.role,
                "input_role": args.role,
                "truth_role": args.role,
                "pretrained_model": str(args.pretrained_model) if args.pretrained_model else None,
                "coordinate_frame": args.coordinate_frame,
                "input_frame": args.input_frame,
                "initial_heading_source": "groundtruth_motion_tangent",
                "initial_heading_distance_m": args.initial_heading_distance_m,
                "truth_max_gap_s": args.truth_max_gap_s,
                "supervision_mode": args.supervision_mode,
                "horizontal_augmentation": not args.disable_horizontal_augmentation,
                "max_gap_s": args.max_gap_s,
                "train_gap_policy": args.train_gap_policy,
                "imu_filter": "causal_moving_average",
                "imu_moving_average_window": args.imu_moving_average_window,
                "imu_moving_average_duration_s": args.imu_moving_average_window / args.sample_rate_hz,
                "sample_rate_hz": args.sample_rate_hz,
                "window_size": args.window_size,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "seed": args.seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    best_loss, best_epoch, stale = math.inf, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for feature, target in train_loader:
            feature, target = feature.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature)
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(feature)
            train_count += len(feature)
        val_loss = evaluate_loss(model, val_loader, device)
        train_loss = train_sum / max(train_count, 1)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"[train] epoch={epoch:02d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_loss,
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "role": args.role,
                    "input_role": args.role,
                    "truth_role": args.role,
                    "data_loader_version": DATA_LOADER_VERSION,
                    "manifest_sha256": data_provenance(args)["manifest_sha256"],
                    "split_sha256": data_provenance(args)["split_sha256"],
                    "input_frame": args.input_frame,
                    "supervision_mode": args.supervision_mode,
                    "dropout": args.dropout,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (args.output_root / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"[train] best_epoch={best_epoch} val_mse={best_loss:.6f}")


def trajectory_prediction(
    model,
    sequence: Sequence,
    window_size: int,
    stride: int,
    device,
    supervision_mode: str,
):
    if supervision_mode == "ronin_window":
        frame_ids = valid_window_starts(sequence, window_size, stride)
    elif supervision_mode == "instantaneous":
        frame_ids = valid_frame_ids(sequence, window_size, stride)
    else:
        raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
    if len(frame_ids) == 0:
        raise ValueError(f"No truth-valid evaluation windows: {sequence.base_id}")
    boundaries = np.flatnonzero(np.diff(frame_ids) != stride) + 1
    runs = np.split(frame_ids, boundaries)
    frame_ids = max(runs, key=len)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame_ids), 2048):
            ids = frame_ids[start:start + 2048]
            if supervision_mode == "ronin_window":
                features = np.stack([sequence.features[index:index + window_size].T for index in ids])
            else:
                features = np.stack([sequence.features[index - window_size:index].T for index in ids])
            predictions.append(model(torch.from_numpy(features).to(device)).cpu().numpy())
    prediction = np.concatenate(predictions)
    prediction = prediction * sequence.target_std + sequence.target_mean
    truth = sequence.position[frame_ids]
    trajectory = np.empty_like(truth)
    trajectory[0] = truth[0]
    trajectory[1:] = trajectory[0] + np.cumsum(prediction[:-1] * sequence.dt_s * stride, axis=0)
    return truth, trajectory, frame_ids


def densify_trajectory(sequence: Sequence, truth: np.ndarray, prediction: np.ndarray, frame_ids: np.ndarray):
    """Interpolate sparse window outputs onto the native 100 Hz evaluation grid.

    RoNIN predicts one displacement per window.  Plotting those endpoints
    directly makes the Vicon path look artificially polygonal (especially for
    2 s windows).  Interpolation is only for visualization/metric sampling; it
    never enters training or checkpoint selection.
    """
    if len(frame_ids) < 2:
        return truth, prediction, frame_ids
    dense_ids = np.arange(int(frame_ids[0]), int(frame_ids[-1]) + 1, dtype=np.int64)
    dense_truth = sequence.position[dense_ids]
    dense_prediction = np.column_stack(
        [np.interp(dense_ids, frame_ids, prediction[:, axis]) for axis in range(prediction.shape[1])]
    ).astype(np.float32)
    return dense_truth.astype(np.float32), dense_prediction, dense_ids


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def velocity_scale_diagnostics(truth: np.ndarray, prediction: np.ndarray, step_dt_s: float):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    truth_path = path_length(truth)
    pred_path = path_length(prediction)
    static_error = np.linalg.norm(np.repeat(truth[:1], len(truth), axis=0) - truth, axis=1)
    if len(truth) < 2:
        return {
            "truth_path_m": truth_path,
            "pred_path_m": pred_path,
            "path_scale_ratio": 0.0,
            "truth_speed_mean_mps": 0.0,
            "pred_speed_mean_mps": 0.0,
            "speed_scale_ratio": 0.0,
            "step_direction_cos_mean": math.nan,
            "step_direction_cos_median": math.nan,
            "static_ate_rmse_m": float(np.sqrt(np.mean(static_error**2))) if len(static_error) else 0.0,
        }

    truth_step = np.diff(truth, axis=0) / max(float(step_dt_s), 1e-6)
    pred_step = np.diff(prediction, axis=0) / max(float(step_dt_s), 1e-6)
    truth_speed = np.linalg.norm(truth_step, axis=1)
    pred_speed = np.linalg.norm(pred_step, axis=1)
    moving = (truth_speed > 0.05) & (pred_speed > 0.05)
    if np.any(moving):
        cosine = np.sum(truth_step[moving] * pred_step[moving], axis=1) / (
            truth_speed[moving] * pred_speed[moving]
        )
        cosine_mean = float(np.mean(cosine))
        cosine_median = float(np.median(cosine))
    else:
        cosine_mean = math.nan
        cosine_median = math.nan
    truth_speed_mean = float(np.mean(truth_speed)) if len(truth_speed) else 0.0
    pred_speed_mean = float(np.mean(pred_speed)) if len(pred_speed) else 0.0
    return {
        "truth_path_m": truth_path,
        "pred_path_m": pred_path,
        "path_scale_ratio": pred_path / max(truth_path, 1e-6),
        "truth_speed_mean_mps": truth_speed_mean,
        "pred_speed_mean_mps": pred_speed_mean,
        "speed_scale_ratio": pred_speed_mean / max(truth_speed_mean, 1e-6),
        "step_direction_cos_mean": cosine_mean,
        "step_direction_cos_median": cosine_median,
        "static_ate_rmse_m": float(np.sqrt(np.mean(static_error**2))),
    }


def evaluate(args, device, specs):
    checkpoint = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    for key in ("role", "input_role", "truth_role"):
        if key in checkpoint and str(checkpoint[key]) != args.role:
            raise ValueError(
                f"Checkpoint {key}={checkpoint[key]} does not match requested role={args.role}"
            )
    expected_checkpoint_provenance = {
        "data_loader_version": DATA_LOADER_VERSION,
        "manifest_sha256": data_provenance(args)["manifest_sha256"],
        "split_sha256": data_provenance(args)["split_sha256"],
    }
    checkpoint_mismatches = [
        field
        for field, value in expected_checkpoint_provenance.items()
        if str(checkpoint.get(field, "")) != value
    ]
    if checkpoint_mismatches:
        raise ValueError(
            "Checkpoint data provenance mismatch: "
            f"{', '.join(checkpoint_mismatches)}; retrain with the manifest-first loader"
        )
    supervision_mode = str(checkpoint.get("supervision_mode", "instantaneous"))
    if "supervision_mode" in checkpoint and supervision_mode != args.supervision_mode:
        print(
            f"[evaluate] using checkpoint supervision_mode={supervision_mode} "
            f"(command-line requested {args.supervision_mode})",
            flush=True,
        )
    input_frame = str(checkpoint.get("input_frame", "ronin_world"))
    if "input_frame" in checkpoint and input_frame != args.input_frame:
        print(
            f"[evaluate] using checkpoint input_frame={input_frame} "
            f"(command-line requested {args.input_frame})",
            flush=True,
        )
    model = build_model(checkpoint["window_size"], float(checkpoint.get("dropout", args.dropout))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = []
    for sequence in read_sequences(args.output_root, "test", args):
        truth, prediction, frame_ids = trajectory_prediction(
            model,
            sequence,
            checkpoint["window_size"],
            checkpoint["stride"],
            device,
            supervision_mode,
        )
        # Evaluate on a continuous native-rate grid. The model still sees only
        # train windows; interpolation never enters checkpoint selection.
        truth, prediction, frame_ids = densify_trajectory(sequence, truth, prediction, frame_ids)
        error = np.linalg.norm(prediction - truth, axis=1)
        row = {
            "base_id": sequence.base_id,
            "split": "test",
            "role": args.role,
            "supervision_mode": supervision_mode,
            "input_frame": input_frame,
            "same_hand": sequence.same_hand,
            "sessions": 1,
            "ate_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "ate_mean_m": float(np.mean(error)),
            "endpoint_error_m": float(error[-1]),
            **velocity_scale_diagnostics(
                truth,
                prediction,
                sequence.dt_s,
            ),
        }
        rows.append(row)
    write_csv(args.output_root / "session_metrics.csv", rows)
    summary = []
    conditions = ("all", "same", "different") if args.role == "mobile" else ("all",)
    for condition in conditions:
        selected = rows if condition == "all" else [row for row in rows if row["same_hand"] == condition]
        if not selected:
            continue
        values = np.asarray([row["ate_rmse_m"] for row in selected])
        summary.append({
            "condition": condition,
            "sessions": len(selected),
            "supervision_mode": supervision_mode,
            "input_frame": input_frame,
            "ate_rmse_mean_m": f"{np.mean(values):.6f}",
            "ate_rmse_median_m": f"{np.median(values):.6f}",
            "endpoint_error_median_m": f"{np.median([row['endpoint_error_m'] for row in selected]):.6f}",
            "path_scale_ratio_median": f"{np.median([row['path_scale_ratio'] for row in selected]):.6f}",
            "speed_scale_ratio_median": f"{np.median([row['speed_scale_ratio'] for row in selected]):.6f}",
            "static_ate_rmse_median_m": f"{np.median([row['static_ate_rmse_m'] for row in selected]):.6f}",
        })
    write_csv(args.output_root / "summary.csv", summary)
    print(f"[evaluate] wrote metrics for {len(rows)} test sessions")


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = read_manifest_specs(args.manifest, args.split_index)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(
        f"[official-ronin] device={device} source={OFFICIAL_SOURCE} "
        f"loader={DATA_LOADER_VERSION} manifest={args.manifest}",
        flush=True,
    )
    if args.command in {"prepare", "all"}:
        prepare(args, specs)
    if args.command in {"train", "all"}:
        train(args, device)
    if args.command in {"evaluate", "all"}:
        evaluate(args, device, specs)


if __name__ == "__main__":
    main()
