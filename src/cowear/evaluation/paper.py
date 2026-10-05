#!/usr/bin/env python3
"""Re-evaluate CoWear trajectories with one median per-session ATE protocol.

This script never trains a model.  It reuses the published Task-1 trajectory
artifacts for PDR/RIDI/RoNIN/TLIO, runs inference from the existing CoWear
LSTM self/common-target checkpoints, and writes auditable per-session metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


from .. import PAPER_ROLE_FOR_STORAGE
from ..models import cowear_lstm
from .._internal import event_pipeline as shared
from ..protocol.checkpoints import read_checkpoint
from ..protocol.geometry import horizontal_ate, information_fusion
from ..protocol.metrics import summarize_ates


ROLES = ("mobile", "watch", "rokid")
ROLE_LABEL = {storage: PAPER_ROLE_FOR_STORAGE[storage].capitalize() for storage in ROLES}
TARGET_DIR = {"mobile": "mobile", "watch": "watch", "rokid": "rokid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split-index",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--self-checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--common-mobile-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--common-watch-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--common-rokid-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def cowear_lstm_args(args: argparse.Namespace, target_mode: str) -> SimpleNamespace:
    """Exact preprocessing configuration used by the completed checkpoints."""
    return SimpleNamespace(
        processed_root=args.processed_root,
        split_index=args.split_index,
        cache_dir=args.cache_dir,
        target_mode=target_mode,
        input_frame="device",
        batch_size=args.batch_size,
        hidden=args.hidden,
        device=args.device,
        seed=args.seed,
        sample_rate_hz=100.0,
        # Match the original paper checkpoints explicitly.
        calibration_mode="published_extrinsic",
        max_gap_s=0.35,
        min_duration_s=4.0,
        smooth_s=0.08,
        gravity_s=0.55,
        truth_max_gap_s=0.02,
        truth_max_jump_deg=45.0,
        truth_max_angular_rate_dps=2000.0,
        truth_max_speed_mps=15.0,
        imu_gap_floor_s=0.03,
        watch_imu_gap_floor_s=0.1,
        imu_gap_median_factor=2.5,
        imu_gap_iqr_factor=3.0,
        imu_gap_ceiling_s=0.15,
        rebuild_cache=False,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def npz_files(root: Path) -> dict[str, Path]:
    return {f"{path.parent.name}/{path.stem}": path for path in root.glob("*/*.npz")}


def trajectory_ate(path: Path) -> tuple[float, int]:
    with np.load(path, allow_pickle=False) as payload:
        truth = payload["truth"].astype(np.float64)
        prediction = payload["prediction"].astype(np.float64)
    count = min(len(truth), len(prediction))
    if count == 0:
        raise ValueError(f"Empty trajectory: {path}")
    truth = truth[:count]
    prediction = prediction[:count]
    if truth.ndim != 2 or truth.shape != prediction.shape or truth.shape[1] < 2:
        raise ValueError(f"trajectory arrays must have at least two coordinates: {path}")
    return horizontal_ate(prediction, truth), count


def contiguous_segments(keys: list[tuple[str, int, int]]) -> list[list[tuple[str, int, int]]]:
    segments: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    previous_end = None
    for key in sorted(keys):
        if previous_end is not None and key[1] != previous_end:
            if current:
                segments.append(current)
            current = []
        current.append(key)
        previous_end = key[2]
    if current:
        segments.append(current)
    return segments


def session_increment_ates(
    values: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float | int]]:
    output = {}
    for session_id in sorted({key[0] for key in values}):
        keys = [key for key in values if key[0] == session_id]
        squared_errors: list[float] = []
        segments = contiguous_segments(keys)
        for segment in segments:
            predicted = np.zeros(3, dtype=np.float64)
            truth = np.zeros(3, dtype=np.float64)
            squared_errors.append(0.0)
            for key in segment:
                mean, _covariance, target = values[key]
                predicted += mean
                truth += target
                error = (predicted - truth)[[0, 2]]
                squared_errors.append(float(error @ error))
        output[session_id] = {
            "ate_m": float(np.sqrt(np.mean(squared_errors))),
            "trajectory_points": len(squared_errors),
            "segments": len(segments),
        }
    return output


def load_model_predictions(
    cfg: SimpleNamespace,
    sessions,
    checkpoint_dir: Path,
    split: str,
    device: torch.device,
    split_watch_hand: bool,
) -> dict[str, dict]:
    tasks = [
        ("mobile", "mobile", "all"),
        ("rokid", "rokid", "all"),
    ]
    if split_watch_hand:
        tasks.extend(
            [("watch_same", "watch", "same"), ("watch_different", "watch", "different")]
        )
    else:
        tasks.append(("watch", "watch", "all"))
    predictions: dict[str, dict] = {}
    for label, role, hand in tasks:
        checkpoint_path = checkpoint_dir / f"{label}.pt"
        checkpoint = read_checkpoint(checkpoint_path)
        model = cowear_lstm.CoWearLSTM(
            checkpoint["mean"], checkpoint["std"], cfg.hidden
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        dataset = cowear_lstm.cache_dataset(cfg, sessions, split, role, hand)
        predictions.setdefault(role, {}).update(
            cowear_lstm.infer(model, dataset, device, cfg.batch_size)
        )
    return predictions


def self_fusion_session_tracks(
    predictions: dict[str, dict],
    weights: dict[str, float] | None,
) -> dict[str, dict]:
    common_keys = set.intersection(*(set(predictions[role]) for role in ROLES))
    output = {}
    for session_id in sorted({key[0] for key in common_keys}):
        keys = [key for key in common_keys if key[0] == session_id]
        # Predictions are already expressed in the calibrated common world
        # frame.  Do not estimate a heading from the test trajectory: that
        # would leak ground-truth motion into the late-fusion protocol.
        squared_errors: list[float] = []
        role_squared = {role: [] for role in ROLES}
        segments = contiguous_segments(keys)
        for segment in segments:
            predicted = {role: np.zeros(3, dtype=np.float64) for role in ROLES}
            target = {role: np.zeros(3, dtype=np.float64) for role in ROLES}
            squared_errors.append(0.0)
            for role in ROLES:
                role_squared[role].append(0.0)
            for key in segment:
                for role in ROLES:
                    mean, _covariance, truth_increment = predictions[role][key]
                    predicted[role] += mean
                    target[role] += truth_increment
                fused = (
                    np.mean(np.stack([predicted[role] for role in ROLES]), axis=0)
                    if weights is None
                    else sum(weights[role] * predicted[role] for role in ROLES)
                )
                phone_truth = target["mobile"]
                error = (fused - phone_truth)[[0, 2]]
                squared_errors.append(float(error @ error))
                for role in ROLES:
                    role_error = (predicted[role] - phone_truth)[[0, 2]]
                    role_squared[role].append(float(role_error @ role_error))
        output[session_id] = {
            "ate_m": float(np.sqrt(np.mean(squared_errors))),
            "trajectory_points": len(squared_errors),
            "segments": len(segments),
            "role_ate_m": {
                role: float(np.sqrt(np.mean(role_squared[role]))) for role in ROLES
            },
        }
    return output


def self_validation_weights(
    validation_predictions: dict[str, dict],
) -> tuple[dict[str, float], dict[str, float]]:
    tracks = self_fusion_session_tracks(
        validation_predictions, weights=None
    )
    median_role_ate = {
        role: float(np.median([values["role_ate_m"][role] for values in tracks.values()]))
        for role in ROLES
    }
    inverse = {role: 1.0 / max(value, 1e-8) for role, value in median_role_ate.items()}
    total = sum(inverse.values())
    return {role: value / total for role, value in inverse.items()}, median_role_ate


def task2_validation_weights(predictions: dict[str, dict]) -> dict[str, float]:
    inverse_errors = {}
    for role, values in predictions.items():
        squared = [
            float(np.sum((mean[[0, 2]] - target[[0, 2]]) ** 2))
            for mean, _covariance, target in values.values()
        ]
        inverse_errors[role] = 1.0 / max(float(np.mean(squared)), 1e-8)
    total = sum(inverse_errors.values())
    return {role: value / total for role, value in inverse_errors.items()}


def task2_increment(
    predictions: dict[str, dict],
    key,
    strategy: str,
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    available = [(role, predictions[role][key]) for role in predictions if key in predictions[role]]
    target = available[0][1][2]
    if strategy == "uniform":
        mean = np.mean([value[0] for _role, value in available], axis=0)
    elif strategy == "fixed_weight":
        active = np.asarray([weights[role] for role, _value in available], dtype=np.float64)
        active /= active.sum()
        mean = sum(weight * value[0] for weight, (_role, value) in zip(active, available))
    elif strategy == "information":
        mean, _covariance = information_fusion(
            np.stack([value[0] for _role, value in available]),
            np.stack([value[1] for _role, value in available]),
        )
    else:
        raise ValueError(strategy)
    return mean, target


def task2_session_ates(
    predictions: dict[str, dict],
    strategy: str,
    weights: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    keys = sorted(set().union(*(set(values) for values in predictions.values())))
    output = {}
    for session_id in sorted({key[0] for key in keys}):
        session_keys = [key for key in keys if key[0] == session_id]
        squared_errors: list[float] = []
        segments = contiguous_segments(session_keys)
        for segment in segments:
            predicted = np.zeros(3, dtype=np.float64)
            truth = np.zeros(3, dtype=np.float64)
            squared_errors.append(0.0)
            for key in segment:
                mean, target = task2_increment(predictions, key, strategy, weights)
                predicted += mean
                truth += target
                error = (predicted - truth)[[0, 2]]
                squared_errors.append(float(error @ error))
        output[session_id] = {
            "ate_m": float(np.sqrt(np.mean(squared_errors))),
            "trajectory_points": len(squared_errors),
            "segments": len(segments),
        }
    return output


def add_metric_rows(
    destination: list[dict],
    task: str,
    method: str,
    target: str,
    estimator: str,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    for session_id, values in sorted(metrics.items()):
        destination.append(
            {
                "task": task,
                "method": method,
                "target": target,
                "estimator": estimator,
                "base_id": session_id,
                "ate_m": values["ate_m"],
                "trajectory_points": values["trajectory_points"],
                "segments": values["segments"],
            }
        )


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["task"], row["method"], row["target"], row["estimator"])
        groups[key].append(float(row["ate_m"]))
    output = []
    for (task, method, target, estimator), values in sorted(groups.items()):
        output.append(
            {
                "task": task,
                "method": method,
                "target": target,
                "estimator": estimator,
                **summarize_ates(values),
            }
        )
    return output


def external_task1_and_fusion(args, per_session_rows: list[dict]) -> None:
    task1_roots = {
        "PDR": {
            role: args.benchmark_root / "pdr_peak_trough" / "trajectories" / "self" / role
            for role in ROLES
        },
        "RIDI": {role: args.benchmark_root / f"ridi_{TARGET_DIR[role]}" / "trajectories" for role in ROLES},
        "RoNIN": {role: args.benchmark_root / f"ronin_{TARGET_DIR[role]}" / "trajectories" for role in ROLES},
        "TLIO": {role: args.benchmark_root / f"tlio_{TARGET_DIR[role]}" / "trajectories" for role in ROLES},
    }
    fusion_roots = {
        "PDR": args.benchmark_root / "pdr_peak_trough" / "trajectories" / "fusion" / "late_fusion_mean",
        "RIDI": args.benchmark_root / "ridi_fusion_uniform" / "trajectories",
        "RoNIN": args.benchmark_root / "ronin_fusion_uniform" / "trajectories",
        "TLIO": args.benchmark_root / "tlio_fusion_uniform" / "trajectories",
    }
    required = []
    for role_root in task1_roots.values():
        required.extend(role_root.values())
    required.extend(fusion_roots.values())
    required.extend([
        args.self_checkpoint_dir,
        args.common_mobile_dir,
        args.common_watch_dir,
        args.common_rokid_dir,
    ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Release asset bundle is incomplete; missing paths:\n- " + "\n- ".join(missing)
        )
    for method, role_roots in task1_roots.items():
        for role, root in role_roots.items():
            files = npz_files(root)
            if not files:
                raise ValueError(f"No {method}/{role} trajectories found")
            if len(files) != 115:
                # Some classical baselines can reject a session at the edge of
                # their validity window. Preserve the measured subset and make
                # the protocol mismatch visible in the run log/table.
                print(
                    f"[paper-eval] warning: {method}/{role} has {len(files)} "
                    "test trajectories (canonical split has 115)",
                    flush=True,
                )
            metrics = {}
            for session_id, path in files.items():
                ate, points = trajectory_ate(path)
                metrics[session_id] = {"ate_m": ate, "trajectory_points": points, "segments": 1}
            add_metric_rows(
                per_session_rows, "task1_self", method, ROLE_LABEL[role], ROLE_LABEL[role], metrics
            )
    for method, root in fusion_roots.items():
        files = npz_files(root)
        if len(files) != 115:
            raise ValueError(f"Expected 115 {method} fusion trajectories, found {len(files)}")
        metrics = {}
        for session_id, path in files.items():
            ate, points = trajectory_ate(path)
            metrics[session_id] = {"ate_m": ate, "trajectory_points": points, "segments": 1}
        add_metric_rows(
            per_session_rows,
            "benchmark_fusion",
            method,
            "Phone",
            "self_checkpoint_uniform",
            metrics,
        )


def checkpoint_manifest(args: argparse.Namespace) -> dict:
    def display(path: Path) -> str:
        try:
            return str(path.relative_to(args.weights_root))
        except ValueError:
            return path.name

    output = {
        "PDR": {"type": "non-learned", "configuration": display(args.benchmark_root / "pdr_peak_trough/parameters.json")},
        "RIDI": {},
        "RoNIN": {},
        "TLIO": {},
        "CoWear LSTM self": {},
        "Task 2 common target": {},
    }
    for role in ROLES:
        output["RIDI"][role] = [
            display(args.benchmark_root / f"ridi_{role}/svr_axis_0.xml"),
            display(args.benchmark_root / f"ridi_{role}/svr_axis_1.xml"),
        ]
        output["RoNIN"][role] = display(args.benchmark_root / f"ronin_{role}/checkpoints/best.pt")
        output["TLIO"][role] = display(args.benchmark_root / f"tlio_{role}/checkpoints/best.pt")
        output["CoWear LSTM self"][role] = display(args.self_checkpoint_dir / f"{role}.pt")
    common_dirs = {
        "mobile": args.common_mobile_dir,
        "watch": args.common_watch_dir,
        "rokid": args.common_rokid_dir,
    }
    for target, root in common_dirs.items():
        output["Task 2 common target"][target] = [
            display(root / "mobile.pt"),
            display(root / "rokid.pt"),
            display(root / "watch_same.pt"),
            display(root / "watch_different.pt"),
        ]
    return output


def main() -> None:
    args = parse_args()
    args.weights_root = args.self_checkpoint_dir.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = cowear_lstm_args(args, "self")
    sessions = shared.load_or_build_sessions(cfg)
    session_map = {session.base_id: session for session in sessions}
    split_counts = {
        split: sum(session.split == split for session in sessions)
        for split in ("train", "val", "test")
    }
    if split_counts["test"] != 115:
        raise ValueError(f"Expected 115 test sessions, found {split_counts['test']}")

    per_session_rows: list[dict] = []
    external_task1_and_fusion(args, per_session_rows)

    # Task 1 CoWear LSTM and benchmark fusion use only the self checkpoints.
    self_val = load_model_predictions(
        cfg, sessions, args.self_checkpoint_dir, "val", device, split_watch_hand=False
    )
    self_test = load_model_predictions(
        cfg, sessions, args.self_checkpoint_dir, "test", device, split_watch_hand=False
    )
    for role in ROLES:
        metrics = session_increment_ates(self_test[role])
        add_metric_rows(
            per_session_rows,
            "task1_self",
            "CoWear LSTM",
            ROLE_LABEL[role],
            ROLE_LABEL[role],
            metrics,
        )

    self_weights, self_validation_median = self_validation_weights(self_val)
    self_uniform = self_fusion_session_tracks(
        self_test, weights=None
    )
    self_weighted = self_fusion_session_tracks(
        self_test, weights=self_weights
    )
    for name, values in (
        ("self_checkpoint_uniform", self_uniform),
        ("self_checkpoint_validation_weighted", self_weighted),
    ):
        metrics = {
            session_id: {
                "ate_m": item["ate_m"],
                "trajectory_points": item["trajectory_points"],
                "segments": item["segments"],
            }
            for session_id, item in values.items()
        }
        add_metric_rows(
            per_session_rows,
            "benchmark_fusion",
            "CoWear LSTM",
            "Phone",
            name,
            metrics,
        )

    # Task 2 remains on its independently trained common-target checkpoints.
    common_dirs = {
        "mobile": args.common_mobile_dir,
        "watch": args.common_watch_dir,
        "rokid": args.common_rokid_dir,
    }
    task2_weights = {}
    for target, checkpoint_dir in common_dirs.items():
        target_cfg = cowear_lstm_args(args, target)
        validation = load_model_predictions(
            target_cfg, sessions, checkpoint_dir, "val", device, split_watch_hand=True
        )
        test = load_model_predictions(
            target_cfg, sessions, checkpoint_dir, "test", device, split_watch_hand=True
        )
        weights = task2_validation_weights(validation)
        task2_weights[target] = weights
        for role in ROLES:
            add_metric_rows(
                per_session_rows,
                "task2_common_target",
                "CoWear LSTM",
                ROLE_LABEL[target],
                f"observation_{ROLE_LABEL[role].lower()}",
                session_increment_ates(test[role]),
            )
        for strategy in ("uniform", "fixed_weight", "information"):
            strategy_metrics = task2_session_ates(test, strategy, weights)
            add_metric_rows(
                per_session_rows,
                "task2_common_target",
                "CoWear LSTM",
                ROLE_LABEL[target],
                f"fusion_{strategy}",
                strategy_metrics,
            )
            if strategy == "information":
                for hand in ("same", "different"):
                    hand_metrics = {
                        session_id: values
                        for session_id, values in strategy_metrics.items()
                        if str(session_map[session_id].same_hand) == hand
                    }
                    add_metric_rows(
                        per_session_rows,
                        "task2_hand_relation",
                        "CoWear LSTM",
                        ROLE_LABEL[target],
                        f"information_{hand}_hand",
                        hand_metrics,
                    )

    summary_rows = summarize_rows(per_session_rows)
    write_csv(args.output_dir / "per_session_ate.csv", per_session_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)

    lookup = {
        (row["task"], row["method"], row["target"], row["estimator"]): row
        for row in summary_rows
    }
    benchmark_rows = []
    benchmark_statistics_rows = []
    for method in ("PDR", "RIDI", "RoNIN", "TLIO", "CoWear LSTM"):
        row = {"method": method}
        statistics_row = {"method": method}
        for role in ROLES:
            item = lookup[("task1_self", method, ROLE_LABEL[role], ROLE_LABEL[role])]
            row[f"{ROLE_LABEL[role].lower()}_median_ate_m"] = item["median_ate_m"]
            for statistic in ("median", "mean", "p25", "p75"):
                statistics_row[
                    f"{ROLE_LABEL[role].lower()}_{statistic}_ate_m"
                ] = item[f"{statistic}_ate_m"]
        fusion_estimator = (
            "self_checkpoint_uniform" if method == "CoWear LSTM" else "self_checkpoint_uniform"
        )
        item = lookup[("benchmark_fusion", method, "Phone", fusion_estimator)]
        row["fusion_median_ate_m"] = item["median_ate_m"]
        for statistic in ("median", "mean", "p25", "p75"):
            statistics_row[f"fusion_{statistic}_ate_m"] = item[f"{statistic}_ate_m"]
        benchmark_rows.append(row)
        benchmark_statistics_rows.append(statistics_row)
    write_csv(args.output_dir / "benchmark_table.csv", benchmark_rows)
    write_csv(
        args.output_dir / "benchmark_table_all_statistics.csv", benchmark_statistics_rows
    )

    task2_rows = [row for row in summary_rows if row["task"] == "task2_common_target"]
    write_csv(args.output_dir / "task2_common_target_table.csv", task2_rows)
    hand_rows = [row for row in summary_rows if row["task"] == "task2_hand_relation"]
    write_csv(args.output_dir / "task2_hand_relation_table.csv", hand_rows)

    phone_self = lookup[("task1_self", "CoWear LSTM", "Phone", "Phone")]["median_ate_m"]
    phone_fusion = lookup[
        ("benchmark_fusion", "CoWear LSTM", "Phone", "self_checkpoint_uniform")
    ]["median_ate_m"]
    report = {
        "metric": {
            "name": "median per-session horizontal ATE",
            "session_definition": "sqrt(mean_t(||prediction-position - truth-position||_xz^2))",
            "aggregate": "median across test sessions; every session has equal aggregate weight",
            "available_session_aggregates": {
                "median_ate_m": "median_s(ATE_s)",
                "mean_ate_m": "mean_s(ATE_s); this is not pooled/global trajectory RMSE",
                "p25_ate_m": "percentile_25_s(ATE_s)",
                "p75_ate_m": "percentile_75_s(ATE_s)",
            },
            "discontinuities": "existing invalid event gaps reset rollout; all valid segment trajectory points remain in the session ATE",
            "pooled_global_rmse_used_as_primary": False,
        },
        "split": {
            "path": args.split_index.name,
            "counts": split_counts,
        },
        "evaluation_script": "cowear.evaluation.paper",
        "checkpoint_manifest": checkpoint_manifest(args),
        "benchmark_fusion_protocol": {
            "inputs": "three Task-1 self-localization checkpoints",
            "steps": [
                "predict each physical device self trajectory",
                "zero origin",
                "retain the calibrated common-world frame (no test-label heading normalization)",
                "intersect the existing joint-event sample grid",
                "uniform arithmetic trajectory mean",
                "evaluate against Phone ground truth",
            ],
            "main_table_strategy": "uniform",
            "weighted_strategy": "validation-only inverse median role ATE",
            "validation_role_median_ate_m": self_validation_median,
            "validation_weights": self_weights,
        },
        "task2_fixed_weights": task2_weights,
        "cowear_lstm_phone": {
            "self_median_ate_m": phone_self,
            "self_checkpoint_uniform_fusion_median_ate_m": phone_fusion,
            "absolute_improvement_m": phone_self - phone_fusion,
            "relative_improvement_percent": 100.0 * (phone_self - phone_fusion) / phone_self,
        },
        "outputs": {
            "per_session": "per_session_ate.csv",
            "all_summaries": "summary.csv",
            "benchmark_table": "benchmark_table.csv",
            "benchmark_all_statistics": "benchmark_table_all_statistics.csv",
            "task2_table": "task2_common_target_table.csv",
            "task2_hand_relation": "task2_hand_relation_table.csv",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
