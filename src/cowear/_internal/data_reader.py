"""Small manifest/JSONL reader shared by the paper baselines.

This module intentionally contains only CoWear I/O.  The former checkout also
carried an unrelated IMUPoser/RIC-Loc trainer; it is not part of the release.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

COWEAR_ROLES = ("mobile", "watch", "rokid")


@dataclass(frozen=True)
class CoWearSessionSpec:
    base_id: str
    date: str
    session_id: str
    split: str
    available_roles: tuple[str, ...]
    watch_primary_file: str
    same_hand: str
    sample_quality: str


def read_cowear_specs(split_index: Path) -> list[CoWearSessionSpec]:
    with split_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    specs = [CoWearSessionSpec(
        base_id=row["base_id"], date=row.get("date", row["base_id"].split("/")[0]),
        session_id=row.get("session_id") or row["base_id"].rsplit("/", 1)[-1],
        split=row["split"], available_roles=tuple(x for x in row["available_roles"].split(",") if x),
        watch_primary_file=row.get("watch_primary_file") or row.get("watch_file", ""),
        same_hand=row.get("same_hand", ""), sample_quality=row.get("sample_quality", ""))
        for row in rows]
    if len(specs) != len({item.base_id for item in specs}):
        raise ValueError(f"duplicate base_id in {split_index}")
    return specs


def sorted_unique(times: list[float], values: list[list[float]], width: int) -> tuple[np.ndarray, np.ndarray]:
    if not times:
        return np.empty(0, dtype=np.float64), np.empty((0, width), dtype=np.float32)
    t, v = np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)
    good = np.isfinite(t) & np.all(np.isfinite(v), axis=1)
    order = np.argsort(t[good], kind="stable")
    t, v = t[good][order], v[good][order]
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], v[keep]


def read_cowear_imu(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    streams: dict[str, tuple[list[float], list[list[float]]]] = {k: ([], []) for k in ("acc", "gyro", "game_quat", "rotation_quat")}
    mapping = {"accelerometer": "acc", "gyroscope": "gyro", "game_rotation_vector": "game_quat", "rotation_vector": "rotation_quat"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line); kind = mapping.get(str(record.get("sensorType", "")).lower())
                timestamp = record.get("alignedRelativeS"); values = record.get("values")
                if kind is None or timestamp is None or not isinstance(values, list) or len(values) < 3:
                    continue
                if kind.endswith("quat"):
                    xyz = [float(value) for value in values[:3]]
                    sample = xyz + [float(values[3])] if len(values) >= 4 else xyz + [math.sqrt(max(0.0, 1.0 - sum(x * x for x in xyz)))]
                else:
                    sample = [float(value) for value in values[:3]]
                streams[kind][0].append(float(timestamp)); streams[kind][1].append(sample)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    widths = {"acc": 3, "gyro": 3, "game_quat": 4, "rotation_quat": 4}
    return {key: sorted_unique(*streams[key], widths[key]) for key in streams}


def cowear_imu_path(session_dir: Path, spec: CoWearSessionSpec, role: str) -> Path | None:
    device_dir = session_dir / "measure" / "align" / role
    names = ([spec.watch_primary_file] if role == "watch" and spec.watch_primary_file else []) + ["imu.jsonl", "imu_relay.jsonl"]
    for name in dict.fromkeys(names):
        path = device_dir / name
        if path.is_file() and path.stat().st_size:
            return path
    return None


def read_cowear_truth(path: Path, role: str) -> tuple[np.ndarray, np.ndarray]:
    times, positions = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device") != role:
                continue
            try:
                times.append(float(row.get("time_s") or float(row["timestamp_ms"]) / 1000.0))
                positions.append([float(row["pos_x_m"]), float(row["pos_z_m"])])
            except (KeyError, TypeError, ValueError):
                continue
    return sorted_unique(times, positions, 2)


def interp_columns(t_src: np.ndarray, values: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_out, t_src, values[:, axis]) for axis in range(values.shape[1])])


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64).copy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0
    return values


def interpolate_quaternion(t_src: np.ndarray, quat: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    if len(t_src) < 2:
        return np.repeat(normalize_quaternions(quat[:1]), len(t_out), axis=0)
    return Slerp(t_src, Rotation.from_quat(normalize_quaternions(quat)))(np.clip(t_out, t_src[0], t_src[-1])).as_quat().astype(np.float32)


def continuous_intervals(timestamps: np.ndarray, max_gap_s: float) -> list[tuple[float, float]]:
    if len(timestamps) < 2:
        return []
    boundaries = np.flatnonzero(np.diff(timestamps) > max_gap_s) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries - 1, len(timestamps) - 1]
    return [(float(timestamps[a]), float(timestamps[b])) for a, b in zip(starts, ends)]


def longest_multistream_overlap(timestamp_streams: Iterable[np.ndarray], max_gap_s: float) -> tuple[float, float] | None:
    overlap: list[tuple[float, float]] | None = None
    for timestamps in timestamp_streams:
        intervals = continuous_intervals(timestamps, max_gap_s)
        if not intervals:
            return None
        if overlap is None:
            overlap = intervals
        else:
            overlap = [(max(a[0], b[0]), min(a[1], b[1])) for a in overlap for b in intervals if min(a[1], b[1]) > max(a[0], b[0])]
        if not overlap:
            return None
    return max(overlap, key=lambda item: item[1] - item[0]) if overlap else None
