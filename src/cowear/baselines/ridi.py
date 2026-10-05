#!/usr/bin/env python3
"""Train and evaluate the RIDI SVR and acceleration-correction baseline on CoWear."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .._internal import classic_data as common


def _cv2():
    """Load OpenCV only when the RIDI SVR backend is actually used."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RIDI requires opencv-python-headless; install the package dependencies"
        ) from exc
    return cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "all"), nargs="?", default="all")
    common.add_data_arguments(parser)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--feature-sigma", type=float, default=1.0)
    parser.add_argument("--target-sigma", type=float, default=15.0)
    parser.add_argument("--max-training-windows", type=int, default=22000)
    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-epsilon", type=float, default=0.01)
    parser.add_argument("--svr-max-iterations", type=int, default=10000)
    parser.add_argument("--correction-lambda", type=float, default=0.1)
    parser.add_argument("--correction-knot-stride", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    common.finish_data_arguments(args)
    if args.output_root is None:
        args.output_root = (
            common.REPO_ROOT / "CoWear" / "results" / "classic_aipdr" / f"ridi_{args.role}"
        )
    return args


def local_velocity(sequence, sigma):
    velocity = np.gradient(sequence.position, sequence.dt_s, axis=0)
    velocity = common.rotate_xy(velocity, -sequence.yaw_rel)
    return gaussian_filter1d(velocity, sigma=sigma, axis=0)


def selected_training_windows(sequences, args):
    candidates = []
    for sequence_id, sequence in enumerate(sequences):
        starts = common.valid_window_starts(sequence, args.window_size, args.stride)
        candidates.extend((sequence_id, int(start)) for start in starts)
    rng = np.random.default_rng(args.seed)
    if len(candidates) > args.max_training_windows:
        selected = rng.choice(len(candidates), size=args.max_training_windows, replace=False)
        candidates = [candidates[index] for index in np.sort(selected)]
    return candidates


def make_training_data(sequences, args):
    selected = selected_training_windows(sequences, args)
    by_sequence = {}
    for row, (sequence_id, start) in enumerate(selected):
        by_sequence.setdefault(sequence_id, []).append((row, start))
    feature_dim = args.window_size * 6
    features = np.empty((len(selected), feature_dim), dtype=np.float32)
    targets = np.empty((len(selected), 2), dtype=np.float32)
    for sequence_id, entries in by_sequence.items():
        sequence = sequences[sequence_id]
        smoothed_features = gaussian_filter1d(
            sequence.ridi_features,
            sigma=args.feature_sigma,
            axis=0,
        )
        smoothed_velocity = local_velocity(sequence, args.target_sigma)
        for row, start in entries:
            end = start + args.window_size
            features[row] = smoothed_features[start:end].reshape(-1)
            targets[row] = smoothed_velocity[end]
    return features, targets


def create_svr(args, feature_dim):
    cv2 = _cv2()
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_EPS_SVR)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setGamma(1.0 / feature_dim)
    model.setC(args.svr_c)
    model.setP(args.svr_epsilon)
    model.setTermCriteria(
        (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, args.svr_max_iterations, 1e-9)
    )
    return model


def train(args):
    sequences = common.read_sequences(args, "train")
    features, targets = make_training_data(sequences, args)
    cv2 = _cv2()
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_paths = []
    for axis in range(2):
        model = create_svr(args, features.shape[1])
        print(f"[ridi] training axis={axis} samples={len(features)} dim={features.shape[1]}", flush=True)
        success = model.train(features, cv2.ml.ROW_SAMPLE, targets[:, axis])
        if not success:
            raise RuntimeError(f"OpenCV SVR training failed for axis {axis}")
        path = args.output_root / f"svr_axis_{axis}.xml"
        model.save(str(path))
        model_paths.append(str(path))
    config = {
        **common.data_provenance(args),
        "method": "RIDI",
        "input": "gravity-stabilized gyroscope + linear acceleration",
        "target": "gravity-stabilized horizontal velocity",
        "single_attachment": "rokid glasses; placement classifier omitted",
        "sample_rate_hz": args.sample_rate_hz,
        "window_size": args.window_size,
        "window_duration_s": args.window_size / args.sample_rate_hz,
        "stride": args.stride,
        "feature_sigma": args.feature_sigma,
        "target_sigma": args.target_sigma,
        "training_windows": len(features),
        "svr_kernel": "RBF",
        "svr_gamma": 1.0 / features.shape[1],
        "svr_c": args.svr_c,
        "svr_epsilon": args.svr_epsilon,
        "correction_lambda": args.correction_lambda,
        "correction_knot_stride": args.correction_knot_stride,
        "correction_frame": "initial-heading horizontal frame",
        "models": model_paths,
        "seed": args.seed,
    }
    (args.output_root / "train_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[ridi] training complete", flush=True)


def load_models(args):
    cv2 = _cv2()
    return [cv2.ml.SVM_load(str(args.output_root / f"svr_axis_{axis}.xml")) for axis in range(2)]


def predict_velocity(sequence, starts, models, args):
    smoothed = gaussian_filter1d(sequence.ridi_features, sigma=args.feature_sigma, axis=0)
    feature = np.stack(
        [smoothed[start:start + args.window_size].reshape(-1) for start in starts]
    ).astype(np.float32)
    local = np.column_stack([model.predict(feature)[1].ravel() for model in models])
    endpoints = starts + args.window_size
    world = common.rotate_xy(local, sequence.yaw_rel[endpoints])
    return endpoints, world


def integrate_regressed_velocity(sequence, endpoints, velocity, stride):
    prediction = np.empty_like(velocity, dtype=np.float64)
    prediction[0] = sequence.position[endpoints[0]]
    dt_s = sequence.dt_s * stride
    if len(prediction) > 1:
        prediction[1:] = prediction[0] + np.cumsum(
            0.5 * (velocity[:-1] + velocity[1:]) * dt_s,
            axis=0,
        )
    return prediction


def linear_bias_basis(length, knot_stride):
    knots = np.arange(0, length, knot_stride, dtype=np.int64)
    if knots[-1] != length - 1:
        knots = np.r_[knots, length - 1]
    basis = np.zeros((length, len(knots)), dtype=np.float64)
    for index in range(length):
        right = int(np.searchsorted(knots, index, side="left"))
        if right == 0:
            basis[index, 0] = 1.0
        elif right == len(knots):
            basis[index, -1] = 1.0
        elif knots[right] == index:
            basis[index, right] = 1.0
        else:
            left = right - 1
            alpha = (index - knots[left]) / (knots[right] - knots[left])
            basis[index, left] = 1.0 - alpha
            basis[index, right] = alpha
    return basis


def integrate_samples(acceleration, initial_velocity, initial_position, dt_s):
    velocity = np.empty_like(acceleration, dtype=np.float64)
    position = np.empty_like(acceleration, dtype=np.float64)
    velocity[0] = initial_velocity
    position[0] = initial_position
    for index in range(1, len(acceleration)):
        velocity[index] = velocity[index - 1] + 0.5 * (
            acceleration[index - 1] + acceleration[index]
        ) * dt_s
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt_s
    return velocity, position


def correct_acceleration(sequence, endpoints, regressed_velocity, args):
    first, last = int(endpoints[0]), int(endpoints[-1])
    acceleration = sequence.world_linear_acc[first:last + 1, :2].astype(np.float64)
    basis = linear_bias_basis(len(acceleration), args.correction_knot_stride)
    raw_velocity, _raw_position = integrate_samples(
        acceleration,
        regressed_velocity[0],
        sequence.position[first],
        sequence.dt_s,
    )
    cumulative_basis = np.zeros_like(basis)
    if len(basis) > 1:
        cumulative_basis[1:] = np.cumsum(
            0.5 * (basis[:-1] + basis[1:]) * sequence.dt_s,
            axis=0,
        )
    offsets = endpoints - first
    design = cumulative_basis[offsets]
    gram = design.T @ design + args.correction_lambda * np.eye(design.shape[1])
    corrected = acceleration.copy()
    for axis in range(2):
        residual = regressed_velocity[:, axis] - raw_velocity[offsets, axis]
        knots = np.linalg.solve(gram, design.T @ residual)
        corrected[:, axis] += basis @ knots
    _velocity, position = integrate_samples(
        corrected,
        regressed_velocity[0],
        sequence.position[first],
        sequence.dt_s,
    )
    return position[offsets]


def validate_config(args):
    config = json.loads((args.output_root / "train_config.json").read_text(encoding="utf-8"))
    expected = common.data_provenance(args)
    mismatch = [key for key, value in expected.items() if config.get(key) != value]
    if mismatch:
        raise ValueError(f"RIDI model provenance mismatch: {', '.join(mismatch)}")


def evaluate(args):
    validate_config(args)
    models = load_models(args)
    corrected_predictions, velocity_predictions = [], []
    for sequence in common.read_sequences(args, args.eval_split):
        starts = common.valid_window_starts(sequence, args.window_size, args.stride)
        starts = common.longest_regular_run(starts, args.stride)
        if not len(starts):
            continue
        endpoints, velocity = predict_velocity(sequence, starts, models, args)
        velocity_position = integrate_regressed_velocity(sequence, endpoints, velocity, args.stride)
        corrected_position = correct_acceleration(sequence, endpoints, velocity, args)
        truth = sequence.position[endpoints]
        velocity_predictions.append((sequence.base_id, truth, velocity_position.astype(np.float32), endpoints))
        corrected_predictions.append((sequence.base_id, truth, corrected_position.astype(np.float32), endpoints))
    evaluation_root = args.evaluation_output_root or args.output_root
    common.write_evaluation(evaluation_root / "velocity_only", "RIDI-SVR velocity", velocity_predictions)
    common.write_evaluation(evaluation_root, "RIDI", corrected_predictions)
    print(f"[ridi] evaluated {len(corrected_predictions)} {args.eval_split} trajectories", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[ridi] loader={common.DATA_LOADER_VERSION}", flush=True)
    if args.command in {"prepare", "all"}:
        common.prepare_cache(args)
    if args.command in {"train", "all"}:
        train(args)
    if args.command in {"evaluate", "all"}:
        evaluate(args)


if __name__ == "__main__":
    main()
