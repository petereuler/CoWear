#!/usr/bin/env python3
"""Single-device CoWear student pose/position calculation and training.

This is intentionally small: it does not do multi-device fusion, reliability,
step segmentation, or long-horizon trajectory evaluation. It only builds the
per-window targets needed by a single-device student network:

The Vicon marker frame and the IMU frame are first related by a per-session
hand-eye calibration ``R_MI``.  The dynamic device pose is then
``R_WI(t) = R_WM(t) R_MI``.  Window targets are expressed in that IMU frame:

* relative pose: q_rel_I = inv(q_WI_start) * q_WI_end
* local displacement: dp_I = R_WI_start.T * (pos_end - pos_start)
* reconstructed world displacement: R_WI_start * dp_I

With ``--epochs > 0`` it trains the student only on these window-level
targets. Long-horizon trajectory integration remains deliberately out of
scope until the single-window pose and position estimates are reliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation

from .data_reader import (
    COWEAR_ROLES,
    CoWearSessionSpec,
    cowear_imu_path,
    interp_columns,
    interpolate_quaternion,
    longest_multistream_overlap,
    normalize_quaternions,
    read_cowear_imu,
    read_cowear_specs,
)


DEFAULT_PROCESSED_ROOT = None
DEFAULT_SPLIT_INDEX = None


@dataclass
class SingleDeviceRecord:
    base_id: str
    split: str
    role: str
    imu_file: str
    # ``t`` is kept near zero for numerically stable per-role interpolation;
    # adding ``time_origin_s`` recovers the shared absolute timestamp clock.
    time_origin_s: float
    t: np.ndarray
    # Original aligned IMU samples are retained so a multi-device consumer can
    # resample them exactly once on its own canonical clock.
    gyro_t_abs_s: np.ndarray
    gyro_device: np.ndarray
    acc_t_abs_s: np.ndarray
    acc_device: np.ndarray
    features: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    q_marker_from_imu: np.ndarray
    calibration: Dict[str, float]


@dataclass
class WindowSample:
    record_idx: int
    start: int
    end: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def moving_average_same(values: np.ndarray, kernel_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    kernel_size = int(max(1, kernel_size))
    if kernel_size <= 1 or len(values) == 0:
        return values.astype(np.float32)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    pad = kernel_size // 2
    if values.ndim == 1:
        padded = np.pad(values, (pad, kernel_size - 1 - pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)
    padded = np.pad(values, ((pad, kernel_size - 1 - pad), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def sorted_pose_rows(times: List[float], values: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not times:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float32)
    good = np.isfinite(time_array) & np.all(np.isfinite(value_array), axis=1)
    time_array = time_array[good]
    value_array = value_array[good]
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    value_array = value_array[order]
    keep = np.r_[True, np.diff(time_array) > 1e-6]
    value_array = value_array[keep]
    return time_array[keep], value_array[:, :3], normalize_quaternions(value_array[:, 3:7]).astype(np.float32)


def read_pose_truth(path: Path, role: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    values: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device") != role:
                continue
            try:
                # ``time_s`` and watch ``alignedRelativeS`` have different
                # origins in the materialized CoWear dataset.  Their absolute
                # aligned timestamps are the shared clock.
                times.append(float(row["timestamp_ms"]) / 1000.0)
                values.append(
                    [
                        float(row["pos_x_m"]),
                        float(row["pos_y_m"]),
                        float(row["pos_z_m"]),
                        float(row["quat_x"]),
                        float(row["quat_y"]),
                        float(row["quat_z"]),
                        float(row["quat_w"]),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted_pose_rows(times, values)


def quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    return out.astype(np.float32)


def quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def quat_angle_deg_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-8, None)
    q2 = q2 / np.clip(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return (2.0 * np.arccos(dot) * 180.0 / math.pi).astype(np.float32)


def quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    out = torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )
    return F.normalize(out, dim=-1, eps=1e-8)


def quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1, eps=1e-8)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def rotate_by_quat_torch(q: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    rot = quat_to_rotmat_torch(q)
    return torch.matmul(rot, vectors.unsqueeze(-1)).squeeze(-1)


def build_features(
    streams: Dict[str, Tuple[np.ndarray, np.ndarray]],
    t: np.ndarray,
    sample_rate_hz: float,
    smooth_s: float,
    include_linear_acc: bool,
    gravity_s: float,
) -> np.ndarray:
    gyro = interp_columns(*streams["gyro"], t).astype(np.float32)
    acc = interp_columns(*streams["acc"], t).astype(np.float32)
    kernel = int(max(1, round(float(smooth_s) * sample_rate_hz)))
    if kernel % 2 == 0:
        kernel += 1
    gyro_f = moving_average_same(gyro, kernel)
    acc_f = moving_average_same(acc, kernel)
    parts = [gyro_f, acc_f]
    if include_linear_acc:
        gravity_kernel = int(max(5, round(float(gravity_s) * sample_rate_hz)))
        if gravity_kernel % 2 == 0:
            gravity_kernel += 1
        gravity = moving_average_same(acc_f, gravity_kernel)
        parts.append((acc_f - gravity).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_marker_from_imu_rotation(source: np.ndarray, target: np.ndarray) -> Rotation:
    rotation, _ = Rotation.align_vectors(target, source)
    residual = np.linalg.norm(rotation.apply(source) - target, axis=1)
    keep = residual <= np.quantile(residual, 0.90)
    if np.count_nonzero(keep) >= 50:
        rotation, _ = Rotation.align_vectors(target[keep], source[keep])
    return rotation


def direction_cosine(source: np.ndarray, target: np.ndarray, rotation: Rotation) -> float:
    predicted = rotation.apply(source)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    valid = denominator > 1e-8
    cosine = np.sum(predicted[valid] * target[valid], axis=1) / denominator[valid]
    return float(np.mean(np.clip(cosine, -1.0, 1.0)))


def calibrate_marker_from_imu(
    gyro_imu: np.ndarray,
    acc_imu: np.ndarray,
    quat_world_marker: np.ndarray,
    sample_rate_hz: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Estimate the fixed R_MI using angular motion and gravity directions."""
    marker_rotation = Rotation.from_quat(quat_world_marker)
    marker_delta = marker_rotation[:-1].inv() * marker_rotation[1:]
    marker_omega = marker_delta.as_rotvec() * float(sample_rate_hz)
    gyro_mid = 0.5 * (gyro_imu[:-1] + gyro_imu[1:])
    angular_kernel = int(max(3, round(0.10 * sample_rate_hz)))
    if angular_kernel % 2 == 0:
        angular_kernel += 1
    marker_omega = moving_average_same(marker_omega, angular_kernel).astype(np.float64)
    gyro_mid = moving_average_same(gyro_mid, angular_kernel).astype(np.float64)
    marker_norm = np.linalg.norm(marker_omega, axis=1)
    gyro_norm = np.linalg.norm(gyro_mid, axis=1)
    active = (
        (marker_norm >= 0.20)
        & (gyro_norm >= 0.20)
        & (marker_norm <= 8.0)
        & (gyro_norm <= 8.0)
    )
    gyro_source = gyro_mid[active]
    gyro_target = marker_omega[active]
    if len(gyro_source) < 100:
        raise ValueError("insufficient angular excitation for marker/IMU calibration")
    gyro_source_unit = gyro_source / np.linalg.norm(gyro_source, axis=1, keepdims=True)
    gyro_target_unit = gyro_target / np.linalg.norm(gyro_target, axis=1, keepdims=True)

    gravity_kernel = int(max(5, round(0.50 * sample_rate_hz)))
    if gravity_kernel % 2 == 0:
        gravity_kernel += 1
    acc_smooth = moving_average_same(acc_imu, gravity_kernel).astype(np.float64)
    acc_unit = acc_smooth / np.clip(np.linalg.norm(acc_smooth, axis=1, keepdims=True), 1e-8, None)
    marker_up = marker_rotation.inv().apply(np.asarray([0.0, 1.0, 0.0]))

    source = np.concatenate([gyro_source_unit, acc_unit, acc_unit], axis=0)
    target = np.concatenate([gyro_target_unit, marker_up, marker_up], axis=0)
    rotation = fit_marker_from_imu_rotation(source, target)

    angular_midpoint = len(gyro_source_unit) // 2
    gravity_midpoint = len(acc_unit) // 2
    first_rotation = fit_marker_from_imu_rotation(
        np.concatenate(
            [
                gyro_source_unit[:angular_midpoint],
                acc_unit[:gravity_midpoint],
                acc_unit[:gravity_midpoint],
            ],
            axis=0,
        ),
        np.concatenate(
            [
                gyro_target_unit[:angular_midpoint],
                marker_up[:gravity_midpoint],
                marker_up[:gravity_midpoint],
            ],
            axis=0,
        ),
    )
    second_rotation = fit_marker_from_imu_rotation(
        np.concatenate(
            [
                gyro_source_unit[angular_midpoint:],
                acc_unit[gravity_midpoint:],
                acc_unit[gravity_midpoint:],
            ],
            axis=0,
        ),
        np.concatenate(
            [
                gyro_target_unit[angular_midpoint:],
                marker_up[gravity_midpoint:],
                marker_up[gravity_midpoint:],
            ],
            axis=0,
        ),
    )
    metrics = {
        "angular_samples": float(len(gyro_source)),
        "gyro_direction_cosine": direction_cosine(gyro_source, gyro_target, rotation),
        "gravity_direction_cosine": direction_cosine(acc_unit, marker_up, rotation),
        "half_rotation_delta_deg": float(
            np.degrees((first_rotation.inv() * second_rotation).magnitude())
        ),
    }
    return rotation.as_quat().astype(np.float32), metrics


def load_published_extrinsic(
    session_dir: Path, role: str
) -> tuple[np.ndarray, Dict[str, float]] | None:
    """Read the released per-session IMU-to-marker extrinsic when available."""
    path = session_dir / "extrinsic_calibration.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entry = payload["roles"][role]
        quaternion = entry.get("quaternion_marker_from_imu_xyzw")
        if quaternion is not None:
            value = np.asarray(quaternion, dtype=np.float64)
            if value.shape != (4,) or not np.all(np.isfinite(value)):
                return None
            value = value / max(float(np.linalg.norm(value)), 1e-12)
            q_marker_from_imu = value.astype(np.float32)
        else:
            matrix = np.asarray(entry["rotation_matrix_marker_from_imu"], dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                return None
            q_marker_from_imu = Rotation.from_matrix(matrix).as_quat().astype(np.float32)
        quality = entry.get("quality", {})
        metrics = {
            "angular_samples": float(quality.get("angular_samples", 0.0)),
            "gyro_direction_cosine": float(quality.get("gyro_direction_cosine", 0.0)),
            "gravity_direction_cosine": float(quality.get("gravity_direction_cosine", 0.0)),
            "half_rotation_delta_deg": float(quality.get("half_rotation_delta_deg", 0.0)),
        }
        return q_marker_from_imu, metrics
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def should_use_spec(
    spec: CoWearSessionSpec, role: str, same_hand_only: bool, hand_group: str = "all"
) -> bool:
    if role not in spec.available_roles:
        return False
    if role == "watch" and same_hand_only and spec.same_hand != "same":
        return False
    if role == "watch" and hand_group != "all" and spec.same_hand != hand_group:
        return False
    return True


def build_record(
    processed_root: Path,
    spec: CoWearSessionSpec,
    role: str,
    sample_rate_hz: float,
    max_gap_s: float,
    min_duration_s: float,
    same_hand_only: bool,
    hand_group: str,
    smooth_s: float,
    include_linear_acc: bool,
    gravity_s: float,
    calibrate_marker_frame: bool,
    require_published_extrinsic: bool = False,
) -> SingleDeviceRecord | None:
    if not should_use_spec(spec, role, same_hand_only, hand_group):
        return None
    session_dir = processed_root / spec.base_id
    truth_path = session_dir / "groundtruth" / "align.csv"
    imu_path = cowear_imu_path(session_dir, spec, role)
    if not truth_path.is_file() or imu_path is None:
        return None
    truth_t, pos, quat = read_pose_truth(truth_path, role)
    if len(truth_t) < 2:
        return None
    streams = read_cowear_imu(imu_path)
    if len(streams["gyro"][0]) < 2 or len(streams["acc"][0]) < 2:
        return None

    info_path = session_dir / "alignment_info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        alignment_info = json.load(handle)
    window_start_s = float(alignment_info["parameters"]["windowStartMs"]) / 1000.0
    streams_abs = {
        name: (values[0] + window_start_s, values[1])
        for name, values in streams.items()
    }

    # Keep interpolation numerically well conditioned after establishing the
    # shared absolute clock.
    common_origin_s = min(
        truth_t[0], streams_abs["gyro"][0][0], streams_abs["acc"][0][0]
    )
    truth_t = truth_t - common_origin_s
    streams = {
        name: (values[0] - common_origin_s, values[1])
        for name, values in streams_abs.items()
    }

    interval = longest_multistream_overlap([truth_t, streams["gyro"][0], streams["acc"][0]], max_gap_s)
    if interval is None:
        return None
    start_s, end_s = interval
    if end_s - start_s < min_duration_s:
        return None
    t = np.arange(start_s, end_s, 1.0 / sample_rate_hz, dtype=np.float64)
    if len(t) < 2:
        return None

    features = build_features(
        streams,
        t,
        sample_rate_hz=sample_rate_hz,
        smooth_s=smooth_s,
        include_linear_acc=include_linear_acc,
        gravity_s=gravity_s,
    )
    pos_i = interp_columns(truth_t, pos, t).astype(np.float32)
    quat_i = interpolate_quaternion(truth_t, quat, t).astype(np.float32)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(pos_i)) or not np.all(np.isfinite(quat_i)):
        return None
    if calibrate_marker_frame:
        published = load_published_extrinsic(session_dir, role)
        if published is not None:
            q_marker_from_imu, calibration = published
        else:
            if require_published_extrinsic:
                raise FileNotFoundError(
                    "published_extrinsic requires the released extrinsic calibration: "
                    f"{session_dir / 'extrinsic_calibration.json'} ({role})"
                )
            try:
                q_marker_from_imu, calibration = calibrate_marker_from_imu(
                    features[:, :3], features[:, 3:6], quat_i, sample_rate_hz
                )
            except (ValueError, np.linalg.LinAlgError):
                return None
    else:
        q_marker_from_imu = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        calibration = {
            "angular_samples": 0.0,
            "gyro_direction_cosine": 0.0,
            "gravity_direction_cosine": 0.0,
            "half_rotation_delta_deg": 0.0,
        }
    q_marker_from_imu_rows = np.broadcast_to(q_marker_from_imu, quat_i.shape)
    quat_world_imu = quat_mul_xyzw(quat_i, q_marker_from_imu_rows)
    return SingleDeviceRecord(
        base_id=spec.base_id,
        split=spec.split,
        role=role,
        imu_file=imu_path.name,
        time_origin_s=float(common_origin_s),
        t=t,
        gyro_t_abs_s=streams_abs["gyro"][0],
        gyro_device=streams_abs["gyro"][1],
        acc_t_abs_s=streams_abs["acc"][0],
        acc_device=streams_abs["acc"][1],
        features=features,
        pos=pos_i,
        quat=quat_world_imu,
        q_marker_from_imu=q_marker_from_imu,
        calibration=calibration,
    )


def build_records(args: argparse.Namespace) -> Dict[str, List[SingleDeviceRecord]]:
    specs = read_cowear_specs(args.split_index)
    by_split: Dict[str, List[SingleDeviceRecord]] = {"train": [], "val": [], "test": []}
    skipped: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for spec in specs:
        record = build_record(
            processed_root=args.processed_root,
            spec=spec,
            role=args.role,
            sample_rate_hz=args.sample_rate_hz,
            max_gap_s=args.max_gap_s,
            min_duration_s=args.min_duration_s,
            same_hand_only=args.same_hand_only,
            hand_group=args.hand_group,
            smooth_s=args.smooth_s,
            include_linear_acc=args.include_linear_acc,
            gravity_s=args.gravity_s,
            calibrate_marker_frame=args.calibrate_marker_frame,
        )
        if record is None:
            if spec.split in skipped and should_use_spec(
                spec, args.role, args.same_hand_only, args.hand_group
            ):
                skipped[spec.split] += 1
            continue
        by_split[record.split].append(record)
    print(
        f"records role={args.role} hand_group={args.hand_group} "
        f"train={len(by_split['train'])} val={len(by_split['val'])} test={len(by_split['test'])} "
        f"skipped={skipped}"
    )
    return by_split


def interval_target(
    record: SingleDeviceRecord, start: int, end: int
) -> Dict[str, np.ndarray]:
    if start < 0 or end <= start or end >= len(record.pos):
        raise ValueError(f"invalid target interval [{start}, {end}] for {record.base_id}")
    pos_start = record.pos[start]
    pos_end = record.pos[end]
    q_start = record.quat[start]
    q_end = record.quat[end]
    dp_world = (pos_end - pos_start).astype(np.float32)
    rot_start = quat_to_rotmat_xyzw(q_start)
    dp_local = (rot_start.T @ dp_world.reshape(3, 1)).reshape(3).astype(np.float32)
    q_rel = quat_mul_xyzw(quat_conj_xyzw(q_start), q_end).astype(np.float32)
    return {
        "pos_start": pos_start.astype(np.float32),
        "pos_end": pos_end.astype(np.float32),
        "q_start": q_start.astype(np.float32),
        "q_end": q_end.astype(np.float32),
        "q_rel": q_rel,
        "dp_world": dp_world,
        "dp_local": dp_local,
    }


class SingleDeviceWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SingleDeviceRecord],
        window: int,
        stride: int,
        target_horizon: int,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        self.records = list(records)
        self.window = int(window)
        self.stride = int(stride)
        self.target_horizon = int(target_horizon)
        if self.target_horizon <= 0 or self.target_horizon >= self.window:
            raise ValueError("target_horizon must be in [1, window-1]")
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.index: List[WindowSample] = []
        for record_idx, record in enumerate(self.records):
            max_start = len(record.features) - self.window
            for start in range(0, max_start + 1, self.stride):
                self.index.append(WindowSample(record_idx=record_idx, start=start, end=start + self.window - 1))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.index[idx]
        record = self.records[sample.record_idx]
        features = torch.tensor(record.features[sample.start : sample.start + self.window], dtype=torch.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean.view(1, -1)) / self.feature_std.view(1, -1)
        target = interval_target(record, sample.end - self.target_horizon, sample.end)
        return {
            "base_id": record.base_id,
            "imu_file": record.imu_file,
            "x": features,
            "pos_start": torch.tensor(target["pos_start"], dtype=torch.float32),
            "pos_end": torch.tensor(target["pos_end"], dtype=torch.float32),
            "q_start": torch.tensor(target["q_start"], dtype=torch.float32),
            "q_end": torch.tensor(target["q_end"], dtype=torch.float32),
            "q_rel": torch.tensor(target["q_rel"], dtype=torch.float32),
            "dp_world": torch.tensor(target["dp_world"], dtype=torch.float32),
            "dp_local": torch.tensor(target["dp_local"], dtype=torch.float32),
        }


def compute_feature_stats(records: Sequence[SingleDeviceRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("no records available for feature stats")
    features = torch.tensor(np.concatenate([record.features for record in records], axis=0), dtype=torch.float32)
    return features.mean(dim=0), features.std(dim=0).clamp_min(1e-5)


class SingleDeviceStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.q_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        q_out = self.q_head[-1]
        nn.init.zeros_(q_out.weight)
        nn.init.zeros_(q_out.bias)
        with torch.no_grad():
            q_out.bias[3] = 1.0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(self.input_norm(x))
        _out, last = self.encoder(h)
        feat = last[-1]
        return {
            "dp_local": self.dp_head(feat),
            "q_rel": F.normalize(self.q_head(feat), dim=-1, eps=1e-8),
        }

    @staticmethod
    def pose_position_from_window(
        pos_start: torch.Tensor,
        q_start: torch.Tensor,
        dp_local: torch.Tensor,
        q_rel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dp_local = dp_local.to(q_start.dtype)
        q_rel = q_rel.to(q_start.dtype)
        dp_world = rotate_by_quat_torch(q_start, dp_local)
        pos_end = pos_start + dp_world
        q_end = quat_mul_torch(q_start, q_rel)
        return {"dp_world": dp_world, "pos_end": pos_end, "q_end": q_end}


def sanity_check_dataset(dataset: SingleDeviceWindowDataset, max_items: int) -> Dict[str, float]:
    count = min(len(dataset), max_items)
    if count == 0:
        raise ValueError("no windows available for sanity check")
    dp_errors: List[float] = []
    pos_errors: List[float] = []
    quat_errors: List[float] = []
    for idx in range(count):
        item = dataset[idx]
        q_start = item["q_start"].numpy()
        q_rel = item["q_rel"].numpy()
        q_end = item["q_end"].numpy()
        dp_local = item["dp_local"].numpy()
        dp_world = item["dp_world"].numpy()
        pos_start = item["pos_start"].numpy()
        pos_end = item["pos_end"].numpy()

        recon_world = (quat_to_rotmat_xyzw(q_start) @ dp_local.reshape(3, 1)).reshape(3)
        recon_pos = pos_start + recon_world
        recon_q_end = quat_mul_xyzw(q_start, q_rel)
        dp_errors.append(float(np.linalg.norm(recon_world - dp_world)))
        pos_errors.append(float(np.linalg.norm(recon_pos - pos_end)))
        quat_errors.append(float(quat_angle_deg_np(recon_q_end[None], q_end[None])[0]))
    return {
        "checked_windows": float(count),
        "dp_world_reconstruction_mean_m": float(np.mean(dp_errors)),
        "dp_world_reconstruction_max_m": float(np.max(dp_errors)),
        "pos_end_reconstruction_mean_m": float(np.mean(pos_errors)),
        "pos_end_reconstruction_max_m": float(np.max(pos_errors)),
        "q_end_reconstruction_mean_deg": float(np.mean(quat_errors)),
        "q_end_reconstruction_max_deg": float(np.max(quat_errors)),
    }


def sanity_check_student_forward(dataset: SingleDeviceWindowDataset, batch_size: int, hidden_dim: int) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    input_dim = int(batch["x"].shape[-1])
    model = SingleDeviceStudent(input_dim=input_dim, hidden_dim=hidden_dim)
    out = model(batch["x"])
    pose_pos = SingleDeviceStudent.pose_position_from_window(
        batch["pos_start"],
        batch["q_start"],
        out["dp_local"],
        out["q_rel"],
    )
    return {
        "input_shape": list(batch["x"].shape),
        "pred_dp_local_shape": list(out["dp_local"].shape),
        "pred_q_rel_shape": list(out["q_rel"].shape),
        "computed_dp_world_shape": list(pose_pos["dp_world"].shape),
        "computed_pos_end_shape": list(pose_pos["pos_end"].shape),
        "computed_q_end_shape": list(pose_pos["q_end"].shape),
    }


def sanity_check_gyro_integration(
    dataset: SingleDeviceWindowDataset, max_items: int
) -> Dict[str, float]:
    count = min(len(dataset), max_items)
    if count == 0:
        raise ValueError("no windows available for gyro integration sanity check")
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    errors: List[float] = []
    for index in indices:
        sample = dataset.index[int(index)]
        record = dataset.records[sample.record_idx]
        target_start = sample.end - dataset.target_horizon
        gyro = record.features[target_start : sample.end, :3]
        integrated = Rotation.identity()
        delta_s = float(np.median(np.diff(record.t[target_start : sample.end + 1])))
        for omega in gyro:
            integrated = integrated * Rotation.from_rotvec(omega * delta_s)
        q_gt = interval_target(record, target_start, sample.end)["q_rel"]
        dot = float(np.clip(abs(np.dot(integrated.as_quat(), q_gt)), 0.0, 1.0))
        errors.append(math.degrees(2.0 * math.acos(dot)))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "checked_windows": float(len(values)),
        "q_rel_error_mean_deg": float(np.mean(values)),
        "q_rel_error_median_deg": float(np.median(values)),
        "q_rel_error_p90_deg": float(np.quantile(values, 0.90)),
    }


def quaternion_alignment_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Quaternion loss invariant to the equivalent q and -q representations."""
    q_pred = F.normalize(q_pred, dim=-1, eps=1e-8)
    q_gt = F.normalize(q_gt, dim=-1, eps=1e-8)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1)).clamp(max=1.0)
    return torch.mean(1.0 - dot)


def student_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor | List[str]],
    orientation_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dp_loss = F.smooth_l1_loss(output["dp_local"], batch["dp_local"])
    orientation_loss = quaternion_alignment_loss(output["q_rel"], batch["q_rel"])
    loss = dp_loss + float(orientation_weight) * orientation_loss
    return loss, dp_loss, orientation_loss


def move_batch_to_device(
    batch: Dict[str, torch.Tensor | List[str]], device: torch.device
) -> Dict[str, torch.Tensor | List[str]]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: SingleDeviceStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    orientation_weight: float,
    grad_clip: float,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "dp_loss": 0.0, "orientation_loss": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["x"])
            loss, dp_loss, orientation_loss = student_loss(output, batch, orientation_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(batch["x"].shape[0])
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["dp_loss"] += float(dp_loss.detach().item()) * batch_size
        totals["orientation_loss"] += float(orientation_loss.detach().item()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_student(
    model: SingleDeviceStudent,
    loader: DataLoader,
    device: torch.device,
    orientation_weight: float,
    use_amp: bool,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    model.eval()
    arrays: Dict[str, List[np.ndarray]] = {
        "dp_pred": [],
        "dp_gt": [],
        "q_pred": [],
        "q_gt": [],
        "pos_pred": [],
        "pos_gt": [],
    }
    total_loss = 0.0
    total_dp_loss = 0.0
    total_orientation_loss = 0.0
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["x"])
            loss, dp_loss, orientation_loss = student_loss(output, batch, orientation_weight)
        pose_position = model.pose_position_from_window(
            batch["pos_start"], batch["q_start"], output["dp_local"], output["q_rel"]
        )
        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_dp_loss += float(dp_loss.item()) * batch_size
        total_orientation_loss += float(orientation_loss.item()) * batch_size
        count += batch_size
        arrays["dp_pred"].append(output["dp_local"].float().cpu().numpy())
        arrays["dp_gt"].append(batch["dp_local"].float().cpu().numpy())
        arrays["q_pred"].append(output["q_rel"].float().cpu().numpy())
        arrays["q_gt"].append(batch["q_rel"].float().cpu().numpy())
        arrays["pos_pred"].append(pose_position["pos_end"].float().cpu().numpy())
        arrays["pos_gt"].append(batch["pos_end"].float().cpu().numpy())

    packed = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    dp_vector_error = np.linalg.norm(packed["dp_pred"] - packed["dp_gt"], axis=1)
    pos_end_error = np.linalg.norm(packed["pos_pred"] - packed["pos_gt"], axis=1)
    q_error = quat_angle_deg_np(packed["q_pred"], packed["q_gt"])
    metrics: Dict[str, object] = {
        "windows": int(count),
        "loss": total_loss / max(count, 1),
        "dp_smooth_l1": total_dp_loss / max(count, 1),
        "orientation_alignment_loss": total_orientation_loss / max(count, 1),
        "dp_local_component_mae_m": np.mean(np.abs(packed["dp_pred"] - packed["dp_gt"]), axis=0).tolist(),
        "dp_local_vector_error_mean_m": float(np.mean(dp_vector_error)),
        "dp_local_vector_error_median_m": float(np.median(dp_vector_error)),
        "dp_local_vector_error_p90_m": float(np.percentile(dp_vector_error, 90)),
        "pos_end_error_mean_m": float(np.mean(pos_end_error)),
        "pos_end_error_median_m": float(np.median(pos_end_error)),
        "pos_end_error_p90_m": float(np.percentile(pos_end_error, 90)),
        "q_rel_error_mean_deg": float(np.mean(q_error)),
        "q_rel_error_median_deg": float(np.median(q_error)),
        "q_rel_error_p90_deg": float(np.percentile(q_error, 90)),
    }
    return metrics, packed


def collect_targets(dataset: SingleDeviceWindowDataset) -> Tuple[np.ndarray, np.ndarray]:
    dp_gt = np.stack([dataset[idx]["dp_local"].numpy() for idx in range(len(dataset))])
    q_gt = np.stack([dataset[idx]["q_rel"].numpy() for idx in range(len(dataset))])
    return dp_gt, q_gt


def baseline_metrics(dataset: SingleDeviceWindowDataset, train_dp_mean: np.ndarray) -> Dict[str, object]:
    dp_gt, q_gt = collect_targets(dataset)
    dp_pred = np.broadcast_to(train_dp_mean.reshape(1, 3), dp_gt.shape)
    q_pred = np.zeros_like(q_gt)
    q_pred[:, 3] = 1.0
    dp_error = np.linalg.norm(dp_pred - dp_gt, axis=1)
    q_error = quat_angle_deg_np(q_pred, q_gt)
    return {
        "definition": "train-mean dp_local and identity q_rel",
        "dp_local_vector_error_mean_m": float(np.mean(dp_error)),
        "dp_local_vector_error_median_m": float(np.median(dp_error)),
        "dp_local_vector_error_p90_m": float(np.percentile(dp_error, 90)),
        "q_rel_error_mean_deg": float(np.mean(q_error)),
        "q_rel_error_median_deg": float(np.median(q_error)),
        "q_rel_error_p90_deg": float(np.percentile(q_error, 90)),
    }


def write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_student(
    args: argparse.Namespace,
    datasets: Dict[str, SingleDeviceWindowDataset],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> Dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
    }
    input_dim = int(feature_mean.numel())
    model = SingleDeviceStudent(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_path = args.output_dir / "best.pt"
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    print(f"training device={device} amp={int(use_amp)} epochs={args.epochs} batch_size={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            args.orientation_weight,
            args.grad_clip,
            scaler,
            use_amp,
        )
        val_metrics, _ = evaluate_student(model, loaders["val"], device, args.orientation_weight, use_amp)
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_dp_loss": train_metrics["dp_loss"],
            "train_orientation_loss": train_metrics["orientation_loss"],
            "val_loss": float(val_metrics["loss"]),
            "val_dp_loss": float(val_metrics["dp_smooth_l1"]),
            "val_orientation_loss": float(val_metrics["orientation_alignment_loss"]),
            "val_pos_end_error_mean_m": float(val_metrics["pos_end_error_mean_m"]),
            "val_q_rel_error_mean_deg": float(val_metrics["q_rel_error_mean_deg"]),
        }
        history.append(row)
        write_history(args.output_dir / "history.csv", history)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_pos={row['val_pos_end_error_mean_m']:.4f}m val_q={row['val_q_rel_error_mean_deg']:.2f}deg "
            f"lr={row['learning_rate']:.2e}"
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _ = evaluate_student(model, loaders["val"], device, args.orientation_weight, use_amp)
    test_metrics, _ = evaluate_student(
        model, loaders["test"], device, args.orientation_weight, use_amp
    )
    train_dp_gt, _ = collect_targets(datasets["train"])
    train_dp_mean = np.mean(train_dp_gt, axis=0)
    baselines = {
        "val": baseline_metrics(datasets["val"], train_dp_mean),
        "test": baseline_metrics(datasets["test"], train_dp_mean),
    }
    report: Dict[str, object] = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation": val_metrics,
        "test": test_metrics,
        "constant_baseline": baselines,
        "checkpoint": str(checkpoint_path),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--split-index", type=Path, default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cowear_single_student_sanity"))
    parser.add_argument("--role", choices=COWEAR_ROLES, default="watch")
    parser.add_argument("--same-hand-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hand-group",
        choices=("all", "same", "different"),
        default="all",
        help="For watch training, select same-hand or different-hand sessions.",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--max-gap-s", type=float, default=0.35)
    parser.add_argument("--min-duration-s", type=float, default=4.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=25,
        help="Target interval in samples; 25 samples at 100 Hz is 0.25 s.",
    )
    parser.add_argument("--smooth-s", type=float, default=0.08)
    parser.add_argument("--include-linear-acc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gravity-s", type=float, default=0.55)
    parser.add_argument(
        "--calibrate-marker-frame", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--sanity-windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--orientation-weight", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(args)
    if not records["train"]:
        raise ValueError("no train records after filtering")
    feature_mean, feature_std = compute_feature_stats(records["train"])
    datasets = {
        split: SingleDeviceWindowDataset(
            split_records,
            args.window,
            args.stride,
            args.target_horizon,
            feature_mean,
            feature_std,
        )
        for split, split_records in records.items()
    }
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"{split} has no windows")

    dataset_stats = {
        split: {
            "records": len(records[split]),
            "windows": len(datasets[split]),
            "input_context_s": float((args.window - 1) / args.sample_rate_hz),
            "target_horizon_s": float(args.target_horizon / args.sample_rate_hz),
            "imu_files": sorted({record.imu_file for record in records[split]}),
            "calibration": {
                key: float(np.median([record.calibration[key] for record in records[split]]))
                for key in (
                    "angular_samples",
                    "gyro_direction_cosine",
                    "gravity_direction_cosine",
                    "half_rotation_delta_deg",
                )
            },
        }
        for split in ("train", "val", "test")
    }
    gt_sanity = {split: sanity_check_dataset(dataset, args.sanity_windows) for split, dataset in datasets.items()}
    gyro_sanity = {
        split: sanity_check_gyro_integration(dataset, args.sanity_windows)
        for split, dataset in datasets.items()
    }
    forward_sanity = sanity_check_student_forward(datasets["train"], args.batch_size, args.hidden_dim)

    report = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "feature_dim": int(feature_mean.numel()),
        "feature_names": ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
        + (["linacc_x", "linacc_y", "linacc_z"] if args.include_linear_acc else []),
        "dataset": dataset_stats,
        "gt_sanity": gt_sanity,
        "gyro_integration_sanity": gyro_sanity,
        "student_forward_sanity": forward_sanity,
    }
    with (args.output_dir / "sanity.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"saved={args.output_dir / 'sanity.json'}")
    if args.epochs > 0:
        training_report = train_student(args, datasets, feature_mean, feature_std)
        print(json.dumps(training_report, indent=2, ensure_ascii=False))
        print(f"saved={args.output_dir / 'metrics.json'}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
