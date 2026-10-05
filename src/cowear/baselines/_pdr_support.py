#!/usr/bin/env python3
"""CoWear session-split PDR and late-fusion baselines.

All collaborative tracks predict the phone's horizontal Vicon trajectory.  The
phone, watch, and glasses PDR tracks are converted to displacement tracks before
late fusion, so their distinct body locations do not become a fixed fusion bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent
from ..protocol import time as ricloc_protocol

REPO_ROOT = SCRIPT_DIR.parents[2]

class _PDRDefaults:
    DEFAULT_THRESHOLDS = {
        "mobile": (0.15, 0.25, 0.40, 0.60, 0.85, 1.15),
        "rokid": (0.15, 0.25, 0.40, 0.60, 0.85, 1.15),
        "watch": (0.20, 0.35, 0.55, 0.80, 1.10, 1.50),
    }

pdr = _PDRDefaults()


@dataclass
class SessionData:
    date: str
    session_id: str
    role: str
    truth_t: np.ndarray
    truth_xy: np.ndarray
    acc_t: np.ndarray
    acc: np.ndarray
    gyro_t: np.ndarray
    gyro: np.ndarray
    quat_t: np.ndarray
    quat: np.ndarray


@dataclass
class Observation:
    data: SessionData
    step_t: np.ndarray
    step_feature: np.ndarray
    heading: np.ndarray
    heading_sign: float
    truth_at_steps: np.ndarray
    interval_distance: np.ndarray


@dataclass
class Calibration:
    fixed_step_m: float
    weinberg_k: float
    train_sessions: int
    train_steps: int


def interp_columns(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, times, values[:, axis]) for axis in range(values.shape[1])])


def truth_at(data: SessionData, query: np.ndarray) -> np.ndarray:
    return interp_columns(data.truth_t, data.truth_xy, query)


def _quaternion_yaw(quat: np.ndarray) -> np.ndarray:
    if len(quat) == 0:
        return np.empty(0)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    x, y, z, w = (quat / np.where(norm > 1e-8, norm, 1.0)).T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _gyro_heading(data: SessionData) -> tuple[np.ndarray, np.ndarray]:
    if len(data.gyro_t) < 2 or len(data.acc_t) < 2:
        return np.empty(0), np.empty(0)
    acc = interp_columns(data.acc_t, data.acc, data.gyro_t)
    vertical = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-6)
    rate = np.sum(data.gyro * vertical, axis=1)
    dt = np.clip(np.diff(data.gyro_t), 0.0, 0.1)
    return data.gyro_t, np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)]


def _initial_truth_heading(data: SessionData, start_t: float, distance_m: float) -> float | None:
    origin = truth_at(data, np.asarray([start_t]))[0]
    index = int(np.searchsorted(data.truth_t, start_t, side="left"))
    for candidate in data.truth_xy[index:]:
        delta = candidate - origin
        if np.linalg.norm(delta) >= distance_m:
            return float(np.arctan2(delta[1], delta[0]))
    return None


def detect_observation(data: SessionData, threshold: float, sample_rate_hz: float,
                       min_step_interval_s: float, initial_heading_distance_m: float,
                       heading_sign: float = 1.0) -> Observation | None:
    if len(data.acc_t) < 100:
        return None
    start = max(float(data.acc_t[0]), float(data.truth_t[0]))
    end = min(float(data.acc_t[-1]), float(data.truth_t[-1]))
    if end - start < 10.0:
        return None
    uniform = np.arange(start, end, 1.0 / sample_rate_hz)
    magnitude = np.linalg.norm(interp_columns(data.acc_t, data.acc, uniform), axis=1)
    try:
        filtered = sosfiltfilt(butter(2, (0.7, 3.0), btype="bandpass", fs=sample_rate_hz, output="sos"), magnitude - np.median(magnitude))
    except ValueError:
        return None
    peaks, _ = find_peaks(filtered, height=threshold, prominence=0.5 * threshold,
                          distance=max(1, round(min_step_interval_s * sample_rate_hz)))
    valleys, _ = find_peaks(-filtered, prominence=0.25 * threshold,
                            distance=max(1, round(0.1 * sample_rate_hz)))
    matched, amplitudes = [], []
    for index in range(1, len(peaks)):
        candidates = valleys[(valleys > peaks[index - 1]) & (valleys < peaks[index])]
        if len(candidates) == 0:
            continue
        trough = int(candidates[np.argmin(filtered[candidates])])
        amplitude = float(filtered[peaks[index]] - filtered[trough])
        if amplitude > 0.0 and np.isfinite(amplitude):
            matched.append(int(peaks[index])); amplitudes.append(amplitude)
    if len(matched) < 8:
        return None
    step_t = uniform[np.asarray(matched)]
    feature = np.maximum(np.asarray(amplitudes), 1e-6) ** 0.25
    heading_t, raw_heading = ((data.quat_t, _quaternion_yaw(data.quat)) if data.role != "watch" and len(data.quat_t) >= 2 else _gyro_heading(data))
    if len(heading_t) < 2:
        return None
    valid = (step_t >= heading_t[0]) & (step_t <= heading_t[-1])
    step_t, feature = step_t[valid], feature[valid]
    if len(step_t) < 8:
        return None
    initial = _initial_truth_heading(data, float(step_t[0]), initial_heading_distance_m)
    if initial is None:
        return None
    sensor = np.interp(step_t, heading_t, raw_heading)
    heading = initial + heading_sign * (sensor - sensor[0])
    truth = truth_at(data, step_t)
    return Observation(data, step_t, feature, heading, heading_sign, truth, np.linalg.norm(np.diff(truth, axis=0), axis=1))


def calibrate(observations: list[Observation]) -> Calibration | None:
    targets, features = [], []
    for observation in observations:
        target, feature = observation.interval_distance, observation.step_feature[1:]
        valid = np.isfinite(target) & np.isfinite(feature) & (target < 2.0)
        if np.count_nonzero(valid) >= 5:
            targets.append(target[valid]); features.append(feature[valid])
    if not targets:
        return None
    target, feature = np.concatenate(targets), np.concatenate(features)
    return Calibration(float(np.mean(target)), float(np.dot(feature, target) / max(np.dot(feature, feature), 1e-12)), len(observations), len(target))


def pdr_positions(observation: Observation, step_lengths: np.ndarray) -> np.ndarray:
    output = np.empty((len(observation.step_t), 2), dtype=np.float64)
    output[0] = observation.truth_at_steps[0]
    increments = np.column_stack((step_lengths[1:] * np.cos(observation.heading[1:]), step_lengths[1:] * np.sin(observation.heading[1:])))
    output[1:] = output[0] + np.cumsum(increments, axis=0)
    return output


pdr.SessionData = SessionData
pdr.Observation = Observation
pdr.Calibration = Calibration
pdr.interp_columns = interp_columns
pdr.truth_at = truth_at
pdr.detect_observation = detect_observation
pdr.calibrate = calibrate
pdr.pdr_positions = pdr_positions

ROLES = ("mobile", "watch", "rokid")
ROLE_LABELS = {"mobile": "phone", "watch": "watch", "rokid": "glasses"}
DEFAULT_THRESHOLDS = pdr.DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class SessionSpec:
    base_id: str
    date: str
    session_id: str
    split: str
    available_roles: tuple[str, ...]
    watch_primary_file: str
    same_hand: str
    sample_quality: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--split-index",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "session_split_seed2027.csv",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "CoWear" / "results" / "pdr")
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-gap-s", type=float, default=0.35)
    parser.add_argument("--min-step-interval-s", type=float, default=0.28)
    parser.add_argument("--initial-heading-distance-m", type=float, default=1.5)
    parser.add_argument(
        "--eval-masks",
        default="all,mobile,watch,rokid,mobile_watch,mobile_rokid,watch_rokid",
        help="Comma-separated retained modality sets; 'all' retains every available role.",
    )
    return parser.parse_args()


def read_specs(split_index: Path) -> list[SessionSpec]:
    with split_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    specs = []
    for row in rows:
        base_id = row["base_id"]
        specs.append(
            SessionSpec(
                base_id=base_id,
                date=row["date"],
                session_id=base_id.rsplit("/", 1)[-1],
                split=row["split"],
                available_roles=tuple(part for part in row["available_roles"].split(",") if part),
                # The aligned export calls this field
                # ``watch_file`` (and also records the full path as
                # ``watch_measure_file``); the older 488-session index used
                # ``watch_primary_file``.
                watch_primary_file=(
                    row.get("watch_primary_file")
                    or row.get("watch_file")
                    or Path(row.get("watch_measure_file", "")).name
                    or "imu.jsonl"
                ),
                same_hand=row["same_hand"],
                sample_quality=row["sample_quality"],
            )
        )
    if len(specs) != len({spec.base_id for spec in specs}):
        raise ValueError(f"Duplicate base_id in {split_index}")
    return specs


def sorted_unique(times: list[float], values: list[list[float]], width: int):
    if not times:
        return np.empty(0, dtype=np.float64), np.empty((0, width), dtype=np.float64)
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    good = np.isfinite(time_array) & np.all(np.isfinite(value_array), axis=1)
    time_array, value_array = time_array[good], value_array[good]
    order = np.argsort(time_array, kind="stable")
    time_array, value_array = time_array[order], value_array[order]
    keep = np.r_[True, np.diff(time_array) > 1e-6]
    return time_array[keep], value_array[keep]


def read_truth(path: Path, role: str):
    times, xy = [], []
    session_dir = ricloc_protocol.session_dir_from_truth_path(path)
    window_start_s = ricloc_protocol.window_start_s(session_dir)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device") != role:
                continue
            try:
                timestamp = float(row["timestamp_ms"]) / 1000.0 - window_start_s
                times.append(timestamp)
                xy.append([float(row["pos_x_m"]), float(row["pos_z_m"])])
            except (KeyError, TypeError, ValueError):
                continue
    return sorted_unique(times, xy, 2)


def read_imu(path: Path):
    streams = {"acc": ([], []), "gyro": ([], []), "game_quat": ([], []), "rotation_quat": ([], [])}
    session_dir = path.parents[3]
    window_start_s = ricloc_protocol.window_start_s(session_dir)
    mapping = {
        "accelerometer": "acc",
        "gyroscope": "gyro",
        "game_rotation_vector": "game_quat",
        "rotation_vector": "rotation_quat",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                stream = mapping.get(str(record.get("sensorType", "")).lower())
                values = record.get("values")
                if stream is None or not isinstance(values, list):
                    continue
                if record.get("alignedRelativeS") is not None:
                    timestamp = float(record["alignedRelativeS"])
                elif record.get("alignedTimestampMs") is not None:
                    timestamp = float(record["alignedTimestampMs"]) / 1000.0 - window_start_s
                else:
                    continue
                width = 4 if stream.endswith("quat") else 3
                if len(values) < width:
                    continue
                streams[stream][0].append(timestamp)
                streams[stream][1].append([float(value) for value in values[:width]])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    acc_t, acc = sorted_unique(*streams["acc"], 3)
    gyro_t, gyro = sorted_unique(*streams["gyro"], 3)
    quat_source = "game_quat" if streams["game_quat"][0] else "rotation_quat"
    quat_t, quat = sorted_unique(*streams[quat_source], 4)
    return acc_t, acc, gyro_t, gyro, quat_t, quat


def load_data(processed_root: Path, spec: SessionSpec, role: str, max_gap_s: float = 0.35):
    if role not in spec.available_roles:
        return None
    session_dir = processed_root / spec.base_id
    truth_t, truth_xy = read_truth(session_dir / "groundtruth" / "align.csv", role)
    imu_name = spec.watch_primary_file if role == "watch" else "imu.jsonl"
    if not imu_name:
        return None
    imu_path = session_dir / "measure" / "align" / role / imu_name
    if not imu_path.is_file() or len(truth_t) < 100:
        return None
    acc_t, acc, gyro_t, gyro, quat_t, quat = read_imu(imu_path)
    if len(acc_t) < 100 or len(gyro_t) < 100:
        return None
    overlap_streams = [truth_t, acc_t, gyro_t]
    if role != "watch" and len(quat_t) >= 2:
        overlap_streams.append(quat_t)
    overlap = ricloc_protocol.longest_multistream_overlap(overlap_streams, max_gap_s=max_gap_s)
    if overlap is None or overlap[1] - overlap[0] < 10.0:
        return None
    start_s, end_s = overlap

    def crop(times: np.ndarray, values: np.ndarray):
        if len(times) == 0:
            return times, values
        selected = (times >= start_s) & (times <= end_s)
        return times[selected], values[selected]

    truth_t, truth_xy = crop(truth_t, truth_xy)
    acc_t, acc = crop(acc_t, acc)
    gyro_t, gyro = crop(gyro_t, gyro)
    quat_t, quat = crop(quat_t, quat)
    return pdr.SessionData(
        date=spec.date,
        session_id=spec.session_id,
        role=role,
        truth_t=truth_t,
        truth_xy=truth_xy,
        acc_t=acc_t,
        acc=acc,
        gyro_t=gyro_t,
        gyro=gyro,
        quat_t=quat_t,
        quat=quat,
    )


def metric(track_t: np.ndarray, pred_xy: np.ndarray, truth_t: np.ndarray, truth_xy: np.ndarray):
    grid = np.arange(float(track_t[0]), float(track_t[-1]) + 1e-6, 1.0)
    truth = pdr.interp_columns(truth_t, truth_xy, grid)
    pred = pdr.interp_columns(track_t, pred_xy, grid)
    error = np.linalg.norm(pred - truth, axis=1)
    path_m = float(np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1)))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(error**2))),
        "ate_mean_m": float(np.mean(error)),
        "ate_median_m": float(np.median(error)),
        "endpoint_error_m": float(error[-1]),
        "endpoint_drift_pct": 100.0 * float(error[-1]) / max(path_m, 1e-6),
        "truth_path_m": path_m,
        "duration_s": float(grid[-1] - grid[0]),
        "grid_t": grid,
        "truth_xy": truth,
        "pred_xy": pred,
    }


def detect(data, parameter: dict, args: argparse.Namespace):
    return pdr.detect_observation(
        data,
        parameter["threshold"],
        args.sample_rate_hz,
        args.min_step_interval_s,
        args.initial_heading_distance_m,
        parameter["heading_sign"],
    )


def tune_role(processed_root: Path, specs: list[SessionSpec], role: str, args: argparse.Namespace):
    candidate_specs = [spec for spec in specs if spec.split in {"train", "val"} and role in spec.available_roles]
    data = [(spec, load_data(processed_root, spec, role, args.max_gap_s)) for spec in candidate_specs]
    data = [(spec, item) for spec, item in data if item is not None]
    best = None
    for threshold in DEFAULT_THRESHOLDS[role]:
        for heading_sign in (1.0, -1.0):
            observations = []
            for spec, item in data:
                observation = pdr.detect_observation(
                    item, threshold, args.sample_rate_hz, args.min_step_interval_s,
                    args.initial_heading_distance_m, heading_sign,
                )
                if observation is not None:
                    observations.append((spec, observation))
            train = [observation for spec, observation in observations if spec.split == "train"]
            val = [observation for spec, observation in observations if spec.split == "val"]
            calibration = pdr.calibrate(train)
            if calibration is None or not val:
                continue
            scores = []
            for observation in val:
                pred = pdr.pdr_positions(observation, calibration.weinberg_k * observation.step_feature)
                scores.append(metric(observation.step_t, pred, observation.data.truth_t, observation.data.truth_xy)["ate_rmse_m"])
            candidate = (float(np.median(scores)), threshold, heading_sign, calibration)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"Could not tune PDR for {role}")
    score, threshold, heading_sign, calibration = best
    return {
        "threshold": threshold,
        "heading_sign": heading_sign,
        "fixed_step_m": calibration.fixed_step_m,
        "weinberg_k": calibration.weinberg_k,
        "validation_median_ate_rmse_m": score,
        "calibration_train_sessions": calibration.train_sessions,
        "calibration_train_steps": calibration.train_steps,
    }


def tune(processed_root: Path, specs: list[SessionSpec], args: argparse.Namespace):
    parameters = {role: tune_role(processed_root, specs, role, args) for role in ROLES}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    return parameters


def parse_masks(raw: str):
    aliases = {
        "all": ROLES,
        "mobile": ("mobile",), "watch": ("watch",), "rokid": ("rokid",),
        "mobile_watch": ("mobile", "watch"),
        "mobile_rokid": ("mobile", "rokid"),
        "watch_rokid": ("watch", "rokid"),
    }
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = set(names) - set(aliases)
    if unknown:
        raise ValueError(f"Unknown evaluation masks: {sorted(unknown)}")
    return {name: aliases[name] for name in names}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]):
    output = []
    for condition in ("all", "same", "different"):
        selected = [row for row in rows if condition == "all" or row["same_hand"] == condition]
        grouped = defaultdict(list)
        for row in selected:
            grouped[(row["split"], row["algorithm"], row["modalities"])].append(row)
        for (split, algorithm, modalities), group in sorted(grouped.items()):
            values = lambda key: np.asarray([float(row[key]) for row in group])
            output.append({
                "split": split, "condition": condition, "algorithm": algorithm,
                "modalities": modalities, "sessions": len(group),
                "ate_rmse_mean_m": f"{np.mean(values('ate_rmse_m')):.6f}",
                "ate_rmse_median_m": f"{np.median(values('ate_rmse_m')):.6f}",
                "endpoint_error_median_m": f"{np.median(values('endpoint_error_m')):.6f}",
            })
    return output


def evaluate(processed_root: Path, specs: list[SessionSpec], parameters: dict, args: argparse.Namespace):
    masks = parse_masks(args.eval_masks)
    role_weight = {role: 1.0 / max(float(value["validation_median_ate_rmse_m"]), 1e-6) for role, value in parameters.items()}
    validation_residuals: dict[str, list[np.ndarray]] = defaultdict(list)
    for spec in specs:
        if spec.split != "val":
            continue
        data = {role: load_data(processed_root, spec, role, args.max_gap_s) for role in ROLES}
        tracks = {}
        for role, item in data.items():
            if item is None:
                continue
            observation = detect(item, parameters[role], args)
            if observation is not None:
                tracks[role] = (
                    observation.step_t,
                    pdr.pdr_positions(
                        observation,
                        parameters[role]["weinberg_k"] * observation.step_feature,
                    ),
                )
        mobile = data.get("mobile")
        if mobile is None or any(role not in tracks for role in ROLES):
            continue
        start = max(float(tracks[role][0][0]) for role in ROLES)
        end = min(float(tracks[role][0][-1]) for role in ROLES)
        if end - start < 10.0:
            continue
        grid = np.arange(start, end + 1e-6, 1.0)
        truth = pdr.truth_at(mobile, grid)
        truth_increment = np.diff(truth, axis=0)
        for role in ROLES:
            track = pdr.interp_columns(tracks[role][0], tracks[role][1], grid)
            validation_residuals[role].append(np.diff(track - track[:1], axis=0) - truth_increment)
    validation_covariance = {
        role: np.cov(np.concatenate(values, axis=0).T) + np.eye(2) * 1e-6
        for role, values in validation_residuals.items()
    }
    rows = []
    for spec in specs:
        if spec.split not in {"val", "test"}:
            continue
        data = {role: load_data(processed_root, spec, role, args.max_gap_s) for role in ROLES}
        tracks = {}
        for role, item in data.items():
            if item is None:
                continue
            observation = detect(item, parameters[role], args)
            if observation is None:
                continue
            pred = pdr.pdr_positions(observation, parameters[role]["weinberg_k"] * observation.step_feature)
            tracks[role] = (observation.step_t, pred)
            result = metric(observation.step_t, pred, item.truth_t, item.truth_xy)
            rows.append({
                "base_id": spec.base_id, "split": spec.split, "same_hand": spec.same_hand,
                "algorithm": "pdr_weinberg", "modalities": role, "target": role,
                **{key: value for key, value in result.items() if key not in {"grid_t", "truth_xy", "pred_xy"}},
            })
        mobile = data.get("mobile")
        if mobile is None:
            continue
        for mask_name, requested_roles in masks.items():
            active = [role for role in requested_roles if role in tracks]
            if len(active) < 2:
                continue
            start = max(float(tracks[role][0][0]) for role in active)
            end = min(float(tracks[role][0][-1]) for role in active)
            if end - start < 10.0:
                continue
            grid = np.arange(start, end + 1e-6, 1.0)
            displacements = []
            for role in active:
                t, pred = tracks[role]
                track = pdr.interp_columns(t, pred, grid)
                displacements.append(track - track[:1])
            truth = pdr.truth_at(mobile, grid)
            strategies = [
                ("late_fusion_mean", np.ones(len(active))),
                ("late_fusion_val_weighted", np.asarray([role_weight[role] for role in active])),
            ]
            if all(role in validation_covariance for role in active):
                strategies.append(("late_fusion_val_information", None))
            for algorithm, weights in strategies:
                if weights is None:
                    information = [np.linalg.inv(validation_covariance[role]) for role in active]
                    fused_covariance = np.linalg.inv(sum(information))
                    fused_displacement = np.stack([
                        fused_covariance @ sum(
                            matrix @ displacements[index][sample]
                            for index, matrix in enumerate(information)
                        )
                        for sample in range(len(grid))
                    ])
                else:
                    weights = weights / weights.sum()
                    fused_displacement = np.tensordot(
                        weights, np.stack(displacements), axes=(0, 0)
                    )
                fused = truth[:1] + fused_displacement
                result = metric(grid, fused, mobile.truth_t, mobile.truth_xy)
                rows.append({
                    "base_id": spec.base_id, "split": spec.split, "same_hand": spec.same_hand,
                    "algorithm": algorithm, "modalities": mask_name, "target": "mobile",
                    **{key: value for key, value in result.items() if key not in {"grid_t", "truth_xy", "pred_xy"}},
                })
    rows.sort(key=lambda row: (row["split"], row["algorithm"], row["modalities"], row["base_id"]))
    write_csv(args.output_root / "session_metrics.csv", rows)
    write_csv(args.output_root / "summary.csv", summarize(rows))
    return rows


def main():
    args = parse_args()
    specs = read_specs(args.split_index)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.command in {"tune", "all"}:
        parameters = tune(args.processed_root, specs, args)
    else:
        parameters = json.loads((args.output_root / "parameters.json").read_text(encoding="utf-8"))
    if args.command in {"evaluate", "all"}:
        rows = evaluate(args.processed_root, specs, parameters, args)
        print(f"wrote {len(rows)} metrics to {args.output_root}")


if __name__ == "__main__":
    main()
