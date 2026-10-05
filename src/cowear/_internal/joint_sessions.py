#!/usr/bin/env python3
"""Joint CoWear step segmentation while retaining dense frame-level data."""
from __future__ import annotations

import argparse, json, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from .device_alignment import (
    build_features,
    build_record,
    quat_mul_xyzw,
    read_pose_truth,
)
from .data_reader import read_cowear_specs

ROLES = ("watch", "mobile", "rokid")

@dataclass
class JointSession:
    base_id: str; split: str; same_hand: str
    # ``t`` remains the backwards-compatible relative clock. ``t_abs_ns`` is
    # the unique canonical cross-device clock and ``t_rel_s`` is its stable
    # floating-point representation for duration calculations.
    t: np.ndarray; t_abs_ns: np.ndarray; t_rel_s: np.ndarray
    features: dict[str, np.ndarray]; raw_features: dict[str, np.ndarray]
    frames: dict[str, np.ndarray]
    pos: dict[str, np.ndarray]
    quat_marker: dict[str, np.ndarray]
    # Backwards-compatible name for the orientation proxy R_WI=R_WM R_MI.
    quat: dict[str, np.ndarray]
    node_valid: dict[str, np.ndarray]; edge_valid: dict[str, np.ndarray]
    truth_position_valid: dict[str, np.ndarray]
    truth_rotation_valid: dict[str, np.ndarray]
    imu_valid: dict[str, np.ndarray]; gravity_valid: dict[str, np.ndarray]
    source_age_s: dict[str, np.ndarray]; bracket_gap_s: dict[str, np.ndarray]
    truth_jump_deg: dict[str, np.ndarray]; truth_speed_mps: dict[str, np.ndarray]
    imu_source_age_s: dict[str, np.ndarray]; imu_bracket_gap_s: dict[str, np.ndarray]
    imu_gap_threshold_s: dict[str, dict[str, float]]
    calibration_quat: dict[str, np.ndarray]
    calibration_provenance: dict[str, dict[str, object]]
    calibration_mode: str
    boundaries: np.ndarray; score: np.ndarray


def slerp_quaternion(t_src: np.ndarray, quat: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Interpolate xyzw quaternions once on the canonical absolute clock."""
    clipped = np.clip(t_out, t_src[0], t_src[-1])
    return Slerp(t_src, Rotation.from_quat(quat))(clipped).as_quat().astype(np.float32)


def interpolation_indices(
    t_src: np.ndarray, t_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve interpolation brackets, treating exact source samples as exact."""
    right = np.searchsorted(t_src, t_out, side="left")
    nearest_right = np.clip(right, 0, len(t_src) - 1)
    nearest_left = np.clip(right - 1, 0, len(t_src) - 1)
    use_right = np.abs(t_src[nearest_right] - t_out) <= np.abs(t_out - t_src[nearest_left])
    nearest = np.where(use_right, nearest_right, nearest_left)
    exact = np.abs(t_src[nearest] - t_out) <= 1e-6
    right = np.clip(right, 1, len(t_src) - 1)
    left = right - 1
    gap = t_src[right] - t_src[left]
    age = np.minimum(np.abs(t_out - t_src[left]), np.abs(t_src[right] - t_out))
    left = np.where(exact, nearest, left)
    right = np.where(exact, nearest, right)
    gap = np.where(exact, 0.0, gap)
    age = np.where(exact, 0.0, age)
    return left, right, exact, age, gap


def truth_interpolation_quality(
    t_src: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    t_out: np.ndarray,
    max_gap_s: float,
    max_jump_deg: float,
    max_angular_rate_dps: float,
    max_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return separate position/rotation validity and interpolation diagnostics."""
    left, right, exact, age, gap = interpolation_indices(t_src, t_out)
    relative = Rotation.from_quat(quat[left]).inv() * Rotation.from_quat(quat[right])
    jump_deg = np.degrees(relative.magnitude())
    distance = np.linalg.norm(pos[right] - pos[left], axis=1)
    rate_denominator = np.maximum(gap, 1e-9)
    angular_rate = np.where(exact, 0.0, jump_deg / rate_denominator)
    speed = np.where(exact, 0.0, distance / rate_denominator)
    if np.any(exact):
        source_index = left[exact]
        previous = np.maximum(source_index - 1, 0)
        following = np.minimum(source_index + 1, len(t_src) - 1)
        gap_previous = t_src[source_index] - t_src[previous]
        gap_following = t_src[following] - t_src[source_index]
        jump_previous = np.degrees(
            (
                Rotation.from_quat(quat[previous]).inv()
                * Rotation.from_quat(quat[source_index])
            ).magnitude()
        )
        jump_following = np.degrees(
            (
                Rotation.from_quat(quat[source_index]).inv()
                * Rotation.from_quat(quat[following])
            ).magnitude()
        )
        speed_previous = np.linalg.norm(pos[source_index] - pos[previous], axis=1) / np.maximum(
            gap_previous, 1e-9
        )
        speed_following = np.linalg.norm(pos[following] - pos[source_index], axis=1) / np.maximum(
            gap_following, 1e-9
        )
        gap[exact] = np.maximum(gap_previous, gap_following)
        jump_deg[exact] = np.maximum(jump_previous, jump_following)
        angular_rate[exact] = np.maximum(
            jump_previous / np.maximum(gap_previous, 1e-9),
            jump_following / np.maximum(gap_following, 1e-9),
        )
        speed[exact] = np.maximum(speed_previous, speed_following)
    in_range = (t_out >= t_src[0]) & (t_out <= t_src[-1])
    gap_valid = gap <= max_gap_s
    position_valid = in_range & gap_valid & (speed <= max_speed_mps)
    rotation_valid = (
        in_range
        & gap_valid
        & (jump_deg <= max_jump_deg)
        & (angular_rate <= max_angular_rate_dps)
    )
    return (
        position_valid,
        rotation_valid,
        age.astype(np.float32),
        gap.astype(np.float32),
        jump_deg.astype(np.float32),
        angular_rate.astype(np.float32),
        speed.astype(np.float32),
    )


def imu_interpolation_quality(
    t_src: np.ndarray,
    t_out: np.ndarray,
    absolute_floor_s: float,
    median_factor: float,
    iqr_factor: float,
    ceiling_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Timestamp-only IMU validity with a rate-adaptive gap threshold."""
    _, _, _, age, gap = interpolation_indices(t_src, t_out)
    right = np.searchsorted(t_src, t_out, side="left")
    nearest = np.clip(right, 0, len(t_src) - 1)
    exact = np.abs(t_src[nearest] - t_out) <= 1e-6
    if np.any(exact):
        source_index = nearest[exact]
        previous = np.maximum(source_index - 1, 0)
        following = np.minimum(source_index + 1, len(t_src) - 1)
        gap[exact] = np.maximum(
            t_src[source_index] - t_src[previous],
            t_src[following] - t_src[source_index],
        )
    positive_dt = np.diff(t_src)
    positive_dt = positive_dt[positive_dt > 0.0]
    nominal = float(np.median(positive_dt))
    q25, q75 = np.percentile(positive_dt, [25, 75])
    # Android streams can be batchy (especially the watch), so the median alone
    # would reject normal 60--85 ms brackets. A Tukey-style upper fence admits
    # the main timing cluster without letting a 5--10% population of large
    # dropouts inflate the threshold. The explicit ceiling is an audit contract.
    threshold = min(
        float(ceiling_s),
        max(
            float(absolute_floor_s),
            float(median_factor) * nominal,
            float(q75 + iqr_factor * (q75 - q25)),
        ),
    )
    valid = (t_out >= t_src[0]) & (t_out <= t_src[-1]) & (gap <= threshold)
    return valid, age.astype(np.float32), gap.astype(np.float32), threshold


def expand_invalid(mask: np.ndarray, width: int) -> np.ndarray:
    """Invalidate filtered samples whose centered support touches an IMU gap."""
    width = max(1, int(width))
    if width % 2 == 0:
        width += 1
    if width == 1 or np.all(mask):
        return mask.copy()
    touched = np.convolve((~mask).astype(np.int16), np.ones(width, np.int16), mode="same") > 0
    return ~touched

def gravity_frame(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate gyro/acc/linear-acc into a frame whose +Z follows gravity.

    Horizontal X is the projection of device X; detection only uses yaw-invariant
    vertical components and horizontal norms.
    """
    gyro, acc, lin = features[:, :3], features[:, 3:6], features[:, 6:9]
    gravity = acc - lin
    z = gravity / np.clip(np.linalg.norm(gravity, axis=1, keepdims=True), 1e-6, None)
    ex = np.broadcast_to(np.array([1., 0., 0.], np.float32), z.shape)
    x = ex - np.sum(ex*z, axis=1, keepdims=True)*z
    bad = np.linalg.norm(x, axis=1) < .1
    ey = np.broadcast_to(np.array([0., 1., 0.], np.float32), z.shape)
    x[bad] = ey[bad] - np.sum(ey[bad]*z[bad], axis=1, keepdims=True)*z[bad]
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-6, None)
    y = np.cross(z, x)
    basis = np.stack((x, y, z), axis=1)
    def rot(v): return np.einsum("nij,nj->ni", basis, v)
    return np.concatenate((rot(gyro), rot(acc), rot(lin)), axis=1).astype(np.float32), basis.astype(np.float32)

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x); scale = 1.4826*np.median(np.abs(x-med))
    return (x-med)/max(float(scale), 1e-4)

def joint_step_boundaries(features: dict[str, np.ndarray], fs: float,
                          raw_features: dict[str, np.ndarray] | None = None,
                          min_period_s=.32, max_period_s=1.5) -> tuple[np.ndarray, np.ndarray]:
    """Detect shared events from gravity-vertical and horizontal-gyro signals.

    ``raw_features`` is accepted for source compatibility but deliberately not
    used. The original paper protocol detects a consensus waveform from the
    gravity-frame vertical channel and horizontal gyro norm for each device.
    """
    sos = butter(2, (.65, 3.0), btype="bandpass", fs=fs, output="sos")
    channels=[]
    for f in features.values():
        vertical = sosfiltfilt(sos, f[:, 8])
        gyro_h = np.linalg.norm(f[:, :2], axis=1)
        gyro_h = sosfiltfilt(sos, gyro_h - np.median(gyro_h))
        channels.append(np.maximum(robust_z(vertical), 0.0) + .35*np.abs(robust_z(gyro_h)))
    stack=np.stack(channels)
    # Median rejects watch-only pendulum peaks; mean term retains weak consensus.
    score=.75*np.median(stack, axis=0)+.25*np.mean(stack, axis=0)
    score=sosfiltfilt(butter(2, 3.2, btype="lowpass", fs=fs, output="sos"), score)
    prom=max(.25, .35*float(np.std(score)))
    peaks,_=find_peaks(score, prominence=prom, distance=max(1, round(min_period_s*fs)))
    if len(peaks)<2: return np.empty((0,2), np.int64), score.astype(np.float32)
    mids=((peaks[:-1]+peaks[1:])//2).astype(np.int64)
    edges=np.r_[max(0, peaks[0]-(mids[0]-peaks[0])), mids,
                min(len(score)-1, peaks[-1]+(peaks[-1]-mids[-1]))]
    pairs=np.stack((edges[:-1], edges[1:]), axis=1)
    duration=(pairs[:,1]-pairs[:,0])/fs
    return pairs[(duration>=min_period_s)&(duration<=max_period_s)], score.astype(np.float32)


def build_sessions(args) -> list[JointSession]:
    calibration_mode = getattr(args, "calibration_mode", "published_extrinsic")
    if calibration_mode not in {"published_extrinsic", "train_role_mean"}:
        raise ValueError(f"unknown calibration mode: {calibration_mode}")
    bundles=[]
    for spec in read_cowear_specs(args.split_index):
        records={}
        for role in ROLES:
            calibrate = calibration_mode == "published_extrinsic" or spec.split == "train"
            r=build_record(args.processed_root,spec,role,args.sample_rate_hz,args.max_gap_s,
                           args.min_duration_s,False,"all",args.smooth_s,True,args.gravity_s,
                           calibrate, calibration_mode == "published_extrinsic")
            if r is None: break
            records[role]=r
        if len(records)==3:
            bundles.append((spec, records))

    role_calibration = {}
    role_calibration_provenance: dict[str, dict[str, object]] = {}
    if calibration_mode == "train_role_mean":
        for role in ROLES:
            candidate_rows = [
                (spec.base_id, records[role].q_marker_from_imu, records[role].calibration)
                for spec, records in bundles
                if spec.split == "train"
            ]
            if not candidate_rows:
                raise ValueError(f"no training calibrations for {role}")
            candidate_quat = np.stack([row[1] for row in candidate_rows])
            mean_rotation = Rotation.from_quat(candidate_quat).mean()
            role_calibration[role] = mean_rotation.as_quat().astype(np.float32)
            residual_deg = np.degrees(
                (mean_rotation.inv() * Rotation.from_quat(candidate_quat)).magnitude()
            )
            quality_summary = {}
            for key in candidate_rows[0][2]:
                values = np.asarray([row[2][key] for row in candidate_rows], dtype=np.float64)
                if "cosine" in key:
                    quality_summary[key] = {
                        "min": float(np.min(values)),
                        "p05": float(np.percentile(values, 5)),
                        "median": float(np.median(values)),
                    }
                elif "delta" in key:
                    quality_summary[key] = {
                        "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                        "max": float(np.max(values)),
                    }
                else:
                    quality_summary[key] = {
                        "min": float(np.min(values)),
                        "p05": float(np.percentile(values, 5)),
                        "median": float(np.median(values)),
                    }
            role_calibration_provenance[role] = {
                "source": "train_split_role_mean",
                "candidate_count": len(candidate_rows),
                "candidate_session_ids": [row[0] for row in candidate_rows],
                "candidate_quat_xyzw": candidate_quat.tolist(),
                "candidate_residual_deg": {
                    "median": float(np.median(residual_deg)),
                    "p95": float(np.percentile(residual_deg, 95)),
                    "max": float(np.max(residual_deg)),
                },
                "candidate_quality": quality_summary,
                "frozen_quat_xyzw": role_calibration[role].tolist(),
            }

    out=[]
    truth_max_gap_s = float(getattr(args, "truth_max_gap_s", 0.02))
    truth_max_jump_deg = float(getattr(args, "truth_max_jump_deg", 45.0))
    truth_max_angular_rate_dps = float(
        getattr(args, "truth_max_angular_rate_dps", 2000.0)
    )
    truth_max_speed_mps = float(getattr(args, "truth_max_speed_mps", 15.0))
    imu_gap_floor_s = float(getattr(args, "imu_gap_floor_s", 0.03))
    watch_imu_gap_floor_s = float(getattr(args, "watch_imu_gap_floor_s", 0.10))
    imu_gap_median_factor = float(getattr(args, "imu_gap_median_factor", 2.5))
    imu_gap_iqr_factor = float(getattr(args, "imu_gap_iqr_factor", 3.0))
    imu_gap_ceiling_s = float(getattr(args, "imu_gap_ceiling_s", 0.15))
    if max(imu_gap_floor_s, watch_imu_gap_floor_s) > imu_gap_ceiling_s:
        raise ValueError("IMU gap floors must not exceed imu_gap_ceiling_s")
    smooth_kernel = int(max(1, round(float(args.smooth_s) * args.sample_rate_hz)))
    gravity_kernel = int(max(5, round(float(args.gravity_s) * args.sample_rate_hz)))
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    if gravity_kernel % 2 == 0:
        gravity_kernel += 1
    filter_support = smooth_kernel + gravity_kernel - 1
    sample_period_ns = int(round(1_000_000_000.0 / float(args.sample_rate_hz)))
    for spec, records in bundles:
        record_times = {
            role: record.t + float(record.time_origin_s)
            for role, record in records.items()
        }
        start=max(values[0] for values in record_times.values())
        end=min(values[-1] for values in record_times.values())
        start_ns = int(np.ceil(start * 1_000_000_000.0 / sample_period_ns)) * sample_period_ns
        end_ns = int(np.floor(end * 1_000_000_000.0 / sample_period_ns)) * sample_period_ns
        t_abs_ns = np.arange(start_ns, end_ns, sample_period_ns, dtype=np.int64)
        if len(t_abs_ns)<400: continue
        t_abs_s = t_abs_ns.astype(np.float64) / 1_000_000_000.0
        t_rel_s = (t_abs_ns - t_abs_ns[0]).astype(np.float64) / 1_000_000_000.0
        feat={}; raw_feat={}; frames={}; pos={}; quat_marker_out={}; quat={}
        node_valid={}; edge_valid={}; source_age_s={}; bracket_gap_s={}
        truth_position_valid={}; truth_rotation_valid={}; truth_jump_deg={}; truth_speed_mps={}
        imu_valid={}; gravity_valid={}; imu_source_age_s={}; imu_bracket_gap_s={}
        imu_gap_threshold_s={}; calibration_quat={}; calibration_provenance={}
        for role,r in records.items():
            role_imu_gap_floor_s = (
                watch_imu_gap_floor_s if role == "watch" else imu_gap_floor_s
            )
            imu_streams = {
                "gyro": (r.gyro_t_abs_s, r.gyro_device),
                "acc": (r.acc_t_abs_s, r.acc_device),
            }
            raw_feat[role] = build_features(
                imu_streams,
                t_abs_s,
                sample_rate_hz=args.sample_rate_hz,
                smooth_s=args.smooth_s,
                include_linear_acc=True,
                gravity_s=args.gravity_s,
            )
            feat[role],frames[role]=gravity_frame(raw_feat[role])
            gyro_ok, gyro_age, gyro_gap, gyro_threshold = imu_interpolation_quality(
                r.gyro_t_abs_s,
                t_abs_s,
                role_imu_gap_floor_s,
                imu_gap_median_factor,
                imu_gap_iqr_factor,
                imu_gap_ceiling_s,
            )
            acc_ok, acc_age, acc_gap, acc_threshold = imu_interpolation_quality(
                r.acc_t_abs_s,
                t_abs_s,
                role_imu_gap_floor_s,
                imu_gap_median_factor,
                imu_gap_iqr_factor,
                imu_gap_ceiling_s,
            )
            # build_features linearly interpolates gyro/acc onto the
            # canonical clock. Internal timestamp gaps are repaired samples;
            # only samples outside both source supports remain invalid.
            if getattr(args, "repair_imu_gaps", True):
                timestamp_valid = (
                    (t_abs_s >= r.gyro_t_abs_s[0])
                    & (t_abs_s <= r.gyro_t_abs_s[-1])
                    & (t_abs_s >= r.acc_t_abs_s[0])
                    & (t_abs_s <= r.acc_t_abs_s[-1])
                )
            else:
                timestamp_valid = expand_invalid(gyro_ok & acc_ok, filter_support)
            gravity_norm = np.linalg.norm(
                raw_feat[role][:, 3:6] - raw_feat[role][:, 6:9], axis=1
            )
            linear_norm = np.linalg.norm(raw_feat[role][:, 6:9], axis=1)
            gravity_valid[role] = (
                np.isfinite(gravity_norm)
                & (gravity_norm >= 3.0)
                & (gravity_norm <= 20.0)
                & np.isfinite(linear_norm)
                & (linear_norm <= 30.0)
            )
            imu_valid[role] = timestamp_valid
            imu_source_age_s[role] = np.maximum(gyro_age, acc_age)
            imu_bracket_gap_s[role] = np.maximum(gyro_gap, acc_gap)
            imu_gap_threshold_s[role] = {
                "gyro": float(gyro_threshold),
                "acc": float(acc_threshold),
            }
            truth_path = args.processed_root / spec.base_id / "groundtruth" / "align.csv"
            truth_t, truth_pos, truth_quat_marker = read_pose_truth(truth_path, role)
            pos[role]=np.stack(
                [np.interp(t_abs_s,truth_t,truth_pos[:,j]) for j in range(3)],1
            ).astype(np.float32)
            quat_marker = slerp_quaternion(truth_t, truth_quat_marker, t_abs_s)
            quat_marker_out[role] = quat_marker
            calibration = (
                records[role].q_marker_from_imu
                if calibration_mode == "published_extrinsic"
                else role_calibration[role]
            )
            calibration_quat[role] = np.asarray(calibration, dtype=np.float32)
            calibration_provenance[role] = (
                {"source": "published_extrinsic", "session_id": spec.base_id,
                 "calibration": records[role].calibration}
                if calibration_mode == "published_extrinsic"
                else role_calibration_provenance[role]
            )
            quat[role] = quat_mul_xyzw(
                quat_marker, np.broadcast_to(calibration_quat[role], quat_marker.shape)
            )
            (
                position_ok,
                rotation_ok,
                age,
                gap,
                jump_deg,
                _angular_rate,
                speed,
            ) = truth_interpolation_quality(
                truth_t,
                truth_pos,
                truth_quat_marker,
                t_abs_s,
                truth_max_gap_s,
                truth_max_jump_deg,
                truth_max_angular_rate_dps,
                truth_max_speed_mps,
            )
            # ``pos`` and ``quat_marker`` are already repaired on the
            # canonical clock: position uses linear interpolation and
            # orientation uses SLERP.  Let repaired truth samples supervise
            # the session when they are inside the source support.  IMU
            # validity follows the repaired canonical-clock support as well;
            # its original bracket gaps remain available for diagnostics.
            if getattr(args, "repair_truth_gaps", True):
                truth_in_range = (t_abs_s >= truth_t[0]) & (t_abs_s <= truth_t[-1])
                position_ok = truth_in_range & np.all(np.isfinite(pos[role]), axis=1)
                rotation_ok = truth_in_range & np.all(np.isfinite(quat_marker), axis=1)
            truth_position_valid[role] = position_ok
            truth_rotation_valid[role] = rotation_ok
            node_valid[role] = position_ok
            task_node_valid = position_ok & rotation_ok & imu_valid[role] & gravity_valid[role]
            edge_valid[role] = task_node_valid[:-1] & task_node_valid[1:]
            source_age_s[role] = age
            bracket_gap_s[role] = gap
            truth_jump_deg[role] = jump_deg
            truth_speed_mps[role] = speed
        bounds,score=joint_step_boundaries(feat,args.sample_rate_hz)
        if len(bounds):
            out.append(JointSession(
                spec.base_id, spec.split, spec.same_hand, t_rel_s, t_abs_ns,
                t_rel_s, feat, raw_feat, frames, pos, quat_marker_out, quat,
                node_valid, edge_valid, truth_position_valid,
                truth_rotation_valid, imu_valid, gravity_valid, source_age_s,
                bracket_gap_s, truth_jump_deg, truth_speed_mps,
                imu_source_age_s, imu_bracket_gap_s, imu_gap_threshold_s,
                calibration_quat, calibration_provenance, calibration_mode,
                bounds, score,
            ))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--processed-root",type=Path,default=Path("data/processed")); p.add_argument("--split-index",type=Path,default=Path("data/splits/session_split_seed2027.csv")); p.add_argument("--output-dir",type=Path,default=Path("outputs/joint_step_audit")); p.add_argument("--sample-rate-hz",type=float,default=100.); p.add_argument("--max-gap-s",type=float,default=.35); p.add_argument("--min-duration-s",type=float,default=4.); p.add_argument("--smooth-s",type=float,default=.08); p.add_argument("--gravity-s",type=float,default=.55); p.add_argument("--truth-max-gap-s",type=float,default=.02); p.add_argument("--truth-max-jump-deg",type=float,default=45.); p.add_argument("--truth-max-angular-rate-dps",type=float,default=2000.); p.add_argument("--truth-max-speed-mps",type=float,default=15.); p.add_argument("--imu-gap-floor-s",type=float,default=.03); p.add_argument("--watch-imu-gap-floor-s",type=float,default=.10); p.add_argument("--imu-gap-median-factor",type=float,default=2.5); p.add_argument("--imu-gap_iqr-factor",type=float,default=3.); p.add_argument("--imu-gap-ceiling-s",type=float,default=.15); p.add_argument("--calibration-mode",choices=("published_extrinsic","train_role_mean"),default="published_extrinsic"); args=p.parse_args()
    random.seed(2027); np.random.seed(2027); args.output_dir.mkdir(parents=True,exist_ok=True)
    sessions=build_sessions(args); audit={"sessions":len(sessions),"events":{},"duration_s":{}}
    for split in ("train","val","test"):
        events=[pair for s in sessions if s.split==split for pair in s.boundaries]
        d=[(int(b)-int(a))/args.sample_rate_hz for a,b in events]
        audit["events"][split]=len(events)
        audit["duration_s"][split]={"median":float(np.median(d)),"p10":float(np.percentile(d,10)),"p90":float(np.percentile(d,90))}
    (args.output_dir/"event_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8"); print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
