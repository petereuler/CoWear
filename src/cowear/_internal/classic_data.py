#!/usr/bin/env python3
"""Shared manifest-first CoWear adapter for classical learned inertial baselines."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from ..protocol import time as ricloc_protocol
from ..evaluation.benchmark import write_csv, write_evaluation

REPO_ROOT = SCRIPT_DIR.parents[2]
from . import ronin_protocol as ronin


DATA_LOADER_VERSION = "cowear_classic_joint_session_v4_initial_pose_gyro_world"
SESSION_CACHE_VERSION = "paper_v1_joint_vertical_gyro_published_extrinsic"
REQUIRED_STREAMS = (
    "accelerometer",
    "gyroscope",
)


@dataclass
class ClassicSequence:
    base_id: str
    split: str
    body_features: np.ndarray
    ridi_features: np.ndarray
    world_features: np.ndarray
    position: np.ndarray
    position3: np.ndarray
    yaw_rel: np.ndarray
    world_linear_acc: np.ndarray
    truth_valid: np.ndarray
    dt_s: float


def add_data_arguments(parser) -> None:
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--split-index",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "session_split_seed2027.csv",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--session-cache",
        type=Path,
        default=None,
        help="Optional aligned JointSession cache; built under --processed-root when absent.",
    )
    parser.add_argument("--role", choices=("mobile", "watch", "rokid"), default="rokid")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--max-gap-s", type=float, default=0.2)
    parser.add_argument("--truth-max-gap-s", type=float, default=0.05)
    parser.add_argument(
        "--truth-time-alignment",
        choices=("manifest", "mobile_vicon_lag"),
        default="manifest",
        help=(
            "Truth timestamp policy. mobile_vicon_lag applies the per-session "
            "residual lag measured from mobile gyro and Vicon angular speed."
        ),
    )
    parser.add_argument(
        "--mobile-vicon-lag-file",
        type=Path,
        default=REPO_ROOT / "CoWear" / "diagnostics" / "mobile_vicon_imu_lag.csv",
    )
    parser.add_argument("--initial-heading-distance-m", type=float, default=1.5)
    parser.add_argument("--min-duration-s", type=float, default=4.0)
    parser.add_argument("--smooth-s", type=float, default=0.08)
    parser.add_argument("--gravity-s", type=float, default=0.55)
    parser.add_argument("--calibration-mode", choices=("published_extrinsic", "train_role_mean"), default="published_extrinsic")
    parser.add_argument("--force-prepare", action="store_true")


def finish_data_arguments(args) -> None:
    if args.manifest is None:
        args.manifest = args.split_index
    if args.session_cache is None:
        args.session_cache = args.processed_root / "joint_sessions.pt"
    if args.cache_root is None:
        args.cache_root = (
            REPO_ROOT / "CoWear" / "work_cache" / f"classic_{args.role}_100hz"
        )


def data_provenance(args) -> dict[str, str]:
    cached = getattr(args, "_classic_data_provenance", None)
    if cached is not None:
        return cached
    truth_time_alignment = getattr(args, "truth_time_alignment", "manifest")
    provenance = {
        "data_loader_version": DATA_LOADER_VERSION,
        "processed_root": str(args.processed_root.resolve()),
        "manifest_source": str(args.manifest.resolve()),
        "manifest_sha256": ronin.file_sha256(args.manifest),
        "split_source": str(args.split_index.resolve()),
        "split_sha256": ronin.file_sha256(args.split_index),
        "time_protocol": ricloc_protocol.TIME_PROTOCOL_VERSION,
        "imu_time_field": "alignedRelativeS + alignment_info.parameters.windowStartMs/1000",
        "truth_time_field": "timestamp_ms/1000",
        "stored_time_origin": "alignment_info.parameters.windowStartMs/1000",
        "role": args.role,
        "target_coordinate_frame": "initial_motion_heading_local",
        "attitude_protocol": "calibrated initial IMU pose plus gyro propagation",
        "session_cache": str(args.session_cache.resolve()),
        "session_cache_mtime_ns": str(args.session_cache.stat().st_mtime_ns),
    }
    if truth_time_alignment != "manifest":
        if truth_time_alignment != "mobile_vicon_lag":
            raise ValueError(f"Unsupported truth-time alignment: {truth_time_alignment}")
        lag_file = getattr(args, "mobile_vicon_lag_file", None)
        if lag_file is None or not lag_file.is_file():
            raise FileNotFoundError(f"Missing mobile Vicon/IMU lag audit: {lag_file}")
        provenance["truth_time_alignment"] = truth_time_alignment
        provenance["mobile_vicon_lag_file"] = str(lag_file.resolve())
        provenance["mobile_vicon_lag_sha256"] = ronin.file_sha256(lag_file)
    args._classic_data_provenance = provenance
    return provenance


def read_vector_streams(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = {name: ([], []) for name in REQUIRED_STREAMS}
    session_dir = path.parents[3]
    window_start_s = ricloc_protocol.window_start_s(session_dir)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                sensor = str(row.get("sensorType", "")).lower()
                vector = row.get("values")
                if sensor not in values or not isinstance(vector, list):
                    continue
                if row.get("alignedRelativeS") is not None:
                    timestamp = float(row["alignedRelativeS"])
                elif row.get("alignedTimestampMs") is not None:
                    timestamp = float(row["alignedTimestampMs"]) / 1000.0 - window_start_s
                else:
                    continue
                width = 4 if sensor == "game_rotation_vector" else 3
                if len(vector) < width:
                    continue
                values[sensor][0].append(timestamp)
                values[sensor][1].append([float(item) for item in vector[:width]])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    output = {}
    for sensor, (times, vectors) in values.items():
        width = 4 if sensor == "game_rotation_vector" else 3
        output[sensor] = ronin.pdr_adapter.sorted_unique(times, vectors, width)
    # The aligned export stores native accelerometer and gyroscope streams.
    # Recreate the AIPDR fallbacks used when Android-derived streams are absent.
    acc_t, acc = output["accelerometer"]
    if len(acc_t) >= 2:
        gravity = np.empty_like(acc)
        window = max(5, int(round(0.5 / np.median(np.diff(acc_t)))))
        window = min(window, len(acc))
        kernel = np.ones(window, dtype=np.float64) / float(window)
        for axis in range(acc.shape[1]):
            gravity[:, axis] = np.convolve(acc[:, axis], kernel, mode="same")
        output["gravity"] = (acc_t.copy(), gravity)
        output["linear_acceleration"] = (acc_t.copy(), acc - gravity)
    return output


def read_truth_pose(path: Path, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, positions, quaternions = [], [], []
    session_dir = ricloc_protocol.session_dir_from_truth_path(path)
    window_start_s = ricloc_protocol.window_start_s(session_dir)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device") != role:
                continue
            try:
                times.append(float(row["timestamp_ms"]) / 1000.0 - window_start_s)
                # Android-style output uses a horizontal x/y plane and z-up.
                positions.append(
                    [float(row["pos_x_m"]), float(row["pos_z_m"]), float(row["pos_y_m"])]
                )
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
    position_t, position = ronin.pdr_adapter.sorted_unique(times, positions, 3)
    quaternion_t, quaternions = ronin.pdr_adapter.sorted_unique(times, quaternions, 4)
    if len(position_t) != len(quaternion_t) or not np.allclose(position_t, quaternion_t):
        quaternions = ronin.interpolate_quaternion(quaternion_t, quaternions, position_t)
    return position_t, position, quaternions


def interpolation_valid(times: np.ndarray, query: np.ndarray, max_gap_s: float) -> np.ndarray:
    if len(times) < 2:
        return np.zeros(len(query), dtype=bool)
    right = np.searchsorted(times, query, side="left")
    right = np.clip(right, 1, len(times) - 1)
    left = right - 1
    return (
        (query >= times[0])
        & (query <= times[-1])
        & ((times[right] - times[left]) <= max_gap_s)
    )


def align_vectors_to_z(values: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Apply the RIDI gravity stabilization while preserving horizontal yaw."""
    gravity = np.asarray(gravity, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(gravity, axis=1)
    source = gravity / np.clip(norms[:, None], 1e-8, None)
    target = np.zeros_like(source)
    target[:, 2] = 1.0
    cross = np.cross(source, target)
    cosine = np.sum(source * target, axis=1)
    sine_sq = np.sum(cross * cross, axis=1)
    output = values.copy()

    regular = sine_sq > 1e-8
    if np.any(regular):
        v = cross[regular]
        x = values[regular]
        first = np.cross(v, x)
        second = np.cross(v, first)
        output[regular] = x + first + second * (
            ((1.0 - cosine[regular]) / sine_sq[regular])[:, None]
        )
    opposite = (~regular) & (cosine < 0.0)
    if np.any(opposite):
        output[opposite, 1:] *= -1.0
    return output


def rotate_xy(values: np.ndarray, angles) -> np.ndarray:
    values = np.asarray(values)
    angles = np.asarray(angles)
    cosine, sine = np.cos(angles), np.sin(angles)
    output = values.copy()
    output[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
    output[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
    return output


def integrated_gravity_yaw(gyro: np.ndarray, gravity: np.ndarray, dt_s: float) -> np.ndarray:
    vertical = gravity / np.clip(np.linalg.norm(gravity, axis=1, keepdims=True), 1e-8, None)
    yaw_rate = np.sum(gyro * vertical, axis=1)
    yaw = np.zeros(len(gyro), dtype=np.float64)
    if len(yaw) > 1:
        yaw[1:] = np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * dt_s)
    return yaw


def cache_path(cache_root: Path, base_id: str, role: str) -> Path:
    return cache_root / "cache" / base_id / f"{role}.npz"


def sequence_sources(args, spec) -> dict[str, str]:
    sources = ronin.aligned_sources(args.processed_root, spec, args.role)
    output = {
        **data_provenance(args),
        "base_id": spec.base_id,
        "imu_source": str(sources["imu_source"].resolve()),
        "truth_source": str(sources["truth_source"].resolve()),
        "alignment_info_source": str(sources["alignment_info_source"].resolve()),
    }
    truth_offset = truth_time_offset(args, spec.base_id, args.role)
    if truth_offset != 0.0:
        output["truth_time_offset_s"] = f"{truth_offset:.6f}"
    return output


def truth_time_offset(args, base_id: str, role: str) -> float:
    return ronin.truth_time_offset(args, base_id, role)


def cache_matches(path: Path, args, spec) -> bool:
    if not path.is_file():
        return False
    expected = sequence_sources(args, spec)
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                math.isclose(float(payload["dt_s"]), 1.0 / args.sample_rate_hz)
                and all(key in payload and str(payload[key]) == value for key, value in expected.items())
            )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def build_sequence(args, spec):
    sources = ronin.validate_aligned_session(args.processed_root, spec, args.role)
    streams = read_vector_streams(sources["imu_source"])
    if any(len(streams[name][0]) < 100 for name in REQUIRED_STREAMS):
        return None
    truth_t, truth_position, _truth_quaternion = read_truth_pose(sources["truth_source"], args.role)
    if len(truth_t) < 100:
        return None
    truth_offset_s = truth_time_offset(args, spec.base_id, args.role)
    truth_t = truth_t - truth_offset_s

    all_times = [streams[name][0] for name in REQUIRED_STREAMS]
    overlap = ricloc_protocol.longest_multistream_overlap(
        [truth_t, *all_times],
        max_gap_s=args.max_gap_s,
    )
    if overlap is None:
        return None
    start, end = overlap
    if end - start < 10.0:
        return None
    dt_s = 1.0 / args.sample_rate_hz
    query = np.arange(start, end, dt_s, dtype=np.float64)
    interpolated = {
        name: ronin.pdr_adapter.pdr.interp_columns(times, values, query)
        for name, (times, values) in streams.items()
        if name != "game_rotation_vector"
    }
    if len(streams.get("game_rotation_vector", ([], []))[0]) >= 2:
        quat_t, quat_values = streams["game_rotation_vector"]
        quat = ronin.interpolate_quaternion(quat_t, quat_values, query)
    else:
        # Gravity-only attitude fallback used by the original AIPDR loader.
        # It resolves tilt while leaving yaw unconstrained (yaw is handled by
        # integrated gyro and the canonical initial-device frame).
        gravity_query = interpolated["gravity"]
        source = gravity_query / np.clip(np.linalg.norm(gravity_query, axis=1, keepdims=True), 1e-8, None)
        target = np.zeros_like(source)
        target[:, 2] = 1.0
        cross = np.cross(source, target)
        w = 1.0 + np.sum(source * target, axis=1)
        quat = np.concatenate((cross, w[:, None]), axis=1)
        opposite = w < 1e-6
        quat[opposite] = np.asarray([1.0, 0.0, 0.0, 0.0])
        quat = ronin.normalize_quaternions(quat)
    acc = interpolated["accelerometer"]
    gyro = interpolated["gyroscope"]
    linear_acc = interpolated["linear_acceleration"]
    gravity = interpolated["gravity"]

    body_features = np.concatenate((gyro, acc), axis=1)
    ridi_features = np.concatenate(
        (align_vectors_to_z(gyro, gravity), align_vectors_to_z(linear_acc, gravity)),
        axis=1,
    )

    world_acc = ronin.rotate_by_quaternion(acc, quat)
    inverse_quat = quat.copy()
    inverse_quat[:, :3] *= -1.0
    inverse_acc = ronin.rotate_by_quaternion(acc, inverse_quat)
    gravity_reference = np.asarray([0.0, 0.0, 9.80665])
    score = lambda item: float(np.sum((np.mean(item, axis=0) - gravity_reference) ** 2))
    if score(inverse_acc) < score(world_acc):
        quat = inverse_quat
        world_acc = inverse_acc
    world_gyro = ronin.rotate_by_quaternion(gyro, quat)
    world_linear_acc = ronin.rotate_by_quaternion(linear_acc, quat)
    initial_imu_yaw = float(ronin.yaw_from_quaternion(quat[:1])[0])
    world_acc = rotate_xy(world_acc, -initial_imu_yaw)
    world_gyro = rotate_xy(world_gyro, -initial_imu_yaw)
    world_linear_acc = rotate_xy(world_linear_acc, -initial_imu_yaw)
    world_features = np.concatenate((world_gyro, world_acc), axis=1)

    position3 = ronin.pdr_adapter.pdr.interp_columns(truth_t, truth_position, query)
    target_heading = ronin.initial_device_yaw(
        sources["truth_source"], args.role, float(query[0]) + truth_offset_s
    )
    if target_heading is None:
        return None
    position3 = position3 - position3[:1]
    position3 = rotate_xy(position3, -target_heading)
    position = position3[:, :2].copy()
    yaw_rel = integrated_gravity_yaw(gyro, gravity, dt_s)

    truth_valid = interpolation_valid(truth_t, query, args.truth_max_gap_s)
    for name in REQUIRED_STREAMS:
        truth_valid &= interpolation_valid(
            streams[name][0],
            query,
            args.max_gap_s,
        )
    arrays = (body_features, ridi_features, world_features, position3, yaw_rel, world_linear_acc)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        return None
    return {
        "body_features": body_features.astype(np.float32),
        "ridi_features": ridi_features.astype(np.float32),
        "world_features": world_features.astype(np.float32),
        "position": position.astype(np.float32),
        "position3": position3.astype(np.float32),
        "yaw_rel": yaw_rel.astype(np.float32),
        "world_linear_acc": world_linear_acc.astype(np.float32),
        "truth_valid": truth_valid,
        "dt_s": np.asarray(dt_s),
    }


def initial_motion_heading(position_xz: np.ndarray, distance_m: float) -> float | None:
    origin = position_xz[0]
    for point in position_xz[1:]:
        delta = point - origin
        if np.linalg.norm(delta) >= distance_m:
            return math.atan2(float(delta[1]), float(delta[0]))
    delta = position_xz[-1] - origin
    if np.linalg.norm(delta) < 0.5:
        return None
    return math.atan2(float(delta[1]), float(delta[0]))


def integrate_calibrated_orientation(session, role: str, sample_rate_hz: float) -> np.ndarray:
    """Propagate W<-I from the calibrated initial pose using only gyroscope data."""
    gyro = session.raw_features[role][:, :3].astype(np.float64)
    orientation = np.empty((len(gyro), 3, 3), dtype=np.float64)
    orientation[0] = Rotation.from_quat(session.quat[role][0]).as_matrix()
    dt_s = 1.0 / sample_rate_hz
    for index in range(len(gyro) - 1):
        orientation[index + 1] = orientation[index] @ Rotation.from_rotvec(
            gyro[index] * dt_s
        ).as_matrix()
    return orientation


def build_sequence_from_joint_session(args, session, role: str):
    """Create every baseline input/target from one canonical aligned session."""
    raw = session.raw_features[role].astype(np.float32)
    gravity_features = session.features[role].astype(np.float32)
    position_world = session.pos[role].astype(np.float64)
    position_xz = position_world[:, (0, 2)]
    heading = initial_motion_heading(position_xz, args.initial_heading_distance_m)
    if heading is None:
        return None

    orientation_wi = integrate_calibrated_orientation(session, role, args.sample_rate_hz)
    gyro_world = np.einsum("nij,nj->ni", orientation_wi, raw[:, :3])
    accel_world = np.einsum("nij,nj->ni", orientation_wi, raw[:, 3:6])
    linear_world = np.einsum("nij,nj->ni", orientation_wi, raw[:, 6:9])

    # Convert Vicon's y-up coordinates to an x-z horizontal navigation frame,
    # then remove the known initial walking heading from both inputs and targets.
    world_gyro_nav = gyro_world[:, (0, 2, 1)]
    world_accel_nav = accel_world[:, (0, 2, 1)]
    world_linear_nav = linear_world[:, (0, 2, 1)]
    world_gyro_nav = rotate_xy(world_gyro_nav, -heading)
    world_accel_nav = rotate_xy(world_accel_nav, -heading)
    world_linear_nav = rotate_xy(world_linear_nav, -heading)

    position3 = position_world[:, (0, 2, 1)] - position_world[:1, (0, 2, 1)]
    position3 = rotate_xy(position3, -heading).astype(np.float32)

    # session.frames stores R_GI.  The heading of R_WG makes RIDI's local
    # velocity target and its inverse rollout transformation exact inverses.
    orientation_wg = np.matmul(
        orientation_wi,
        session.frames[role].astype(np.float64).transpose(0, 2, 1),
    )
    gravity_x_world = orientation_wg[:, :, 0]
    gravity_heading = np.unwrap(
        np.arctan2(gravity_x_world[:, 2], gravity_x_world[:, 0])
    ) - heading

    edge = session.edge_valid[role].astype(bool)
    truth_valid = np.zeros(len(raw), dtype=bool)
    if len(edge):
        truth_valid[0] = edge[0]
        truth_valid[1:] = edge

    arrays = (
        raw,
        gravity_features,
        world_gyro_nav,
        world_accel_nav,
        world_linear_nav,
        position3,
        gravity_heading,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        return None
    return {
        "body_features": raw[:, :6].astype(np.float32),
        "ridi_features": np.concatenate(
            (gravity_features[:, :3], gravity_features[:, 6:9]), axis=1
        ).astype(np.float32),
        "world_features": np.concatenate((world_gyro_nav, world_accel_nav), axis=1).astype(np.float32),
        "position": position3[:, :2].astype(np.float32),
        "position3": position3,
        "yaw_rel": gravity_heading.astype(np.float32),
        "world_linear_acc": world_linear_nav.astype(np.float32),
        "truth_valid": truth_valid,
        "dt_s": np.asarray(1.0 / args.sample_rate_hz),
    }


def prepare_cache(args) -> list[dict[str, str]]:
    specs = ronin.read_manifest_specs(args.manifest, args.split_index)
    # Import the defining module before unpickling the portable JointSession
    # objects.  All baselines then consume exactly the same aligned samples,
    # validity masks, calibration, and coordinate contract as the reference.
    from . import joint_sessions  # noqa: F401

    cache_payload = None
    if args.session_cache.is_file():
        cache_payload = torch.load(args.session_cache, map_location="cpu", weights_only=False)
    cache_is_current = (
        isinstance(cache_payload, dict)
        and cache_payload.get("session_cache_version") == SESSION_CACHE_VERSION
        and cache_payload.get("calibration_mode") == args.calibration_mode
        and isinstance(cache_payload.get("sessions"), list)
    )
    if not cache_is_current:
        from .joint_sessions import build_sessions

        sessions = build_sessions(args)
        args.session_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"session_cache_version": SESSION_CACHE_VERSION, "calibration_mode": args.calibration_mode, "sessions": sessions}, args.session_cache)
        cache_payload = {"session_cache_version": SESSION_CACHE_VERSION, "calibration_mode": args.calibration_mode, "sessions": sessions}
    session_payload = cache_payload
    sessions = session_payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError(f"Invalid JointSession cache: {args.session_cache}")
    sessions_by_id = {session.base_id: session for session in sessions}
    rows = []
    for index, spec in enumerate(specs, 1):
        if args.role not in spec.available_roles:
            continue
        session = sessions_by_id.get(spec.base_id)
        if session is None:
            continue
        path = cache_path(args.cache_root, spec.base_id, args.role)
        if args.force_prepare or not cache_matches(path, args, spec):
            payload = build_sequence_from_joint_session(args, session, args.role)
            if payload is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = sequence_sources(args, spec)
            np.savez_compressed(
                path,
                **payload,
                **{key: np.asarray(value) for key, value in metadata.items()},
            )
        with np.load(path, allow_pickle=False) as payload:
            rows.append(
                {
                    "base_id": spec.base_id,
                    "split": spec.split,
                    "role": args.role,
                    "cache_path": str(path.resolve()),
                    "samples": str(len(payload["position"])),
                    "valid_samples": str(int(np.count_nonzero(payload["truth_valid"]))),
                    "imu_source": str(payload["imu_source"]),
                    "truth_source": str(payload["truth_source"]),
                }
            )
        if index % 50 == 0 or index == len(specs):
            print(f"[prepare-classic] {index}/{len(specs)}", flush=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.cache_root / "cache_index.csv", rows)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ronin.VALID_SPLITS}
    (args.cache_root / "data_provenance.json").write_text(
        json.dumps(
            {
                **data_provenance(args),
                "selected_sessions": len(rows),
                "selected_sessions_by_split": split_counts,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return rows


def read_sequences(args, split: str) -> list[ClassicSequence]:
    index_path = args.cache_root / "cache_index.csv"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    sequences = []
    expected_global = data_provenance(args)
    for row in rows:
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            mismatches = [
                key
                for key, value in expected_global.items()
                if key not in payload or str(payload[key]) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Classic cache provenance mismatch in {row['cache_path']}: {', '.join(mismatches)}"
                )
            sequences.append(
                ClassicSequence(
                    base_id=row["base_id"],
                    split=row["split"],
                    body_features=payload["body_features"],
                    ridi_features=payload["ridi_features"],
                    world_features=payload["world_features"],
                    position=payload["position"],
                    position3=payload["position3"],
                    yaw_rel=payload["yaw_rel"],
                    world_linear_acc=payload["world_linear_acc"],
                    truth_valid=payload["truth_valid"].astype(bool),
                    dt_s=float(payload["dt_s"]),
                )
            )
    return sequences


def valid_window_starts(sequence: ClassicSequence, window_size: int, stride: int) -> np.ndarray:
    invalid_prefix = np.r_[0, np.cumsum(~sequence.truth_valid)]
    starts = []
    for start in range(0, len(sequence.position) - window_size, stride):
        if invalid_prefix[start + window_size + 1] - invalid_prefix[start] == 0:
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def longest_regular_run(starts: np.ndarray, stride: int) -> np.ndarray:
    if not len(starts):
        return starts
    breaks = np.flatnonzero(np.diff(starts) != stride) + 1
    return max(np.split(starts, breaks), key=len)
