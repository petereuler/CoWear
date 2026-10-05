#!/usr/bin/env python3
"""Train the official TLIO ResNet displacement network on aligned CoWear data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._internal import classic_data as common
from ..training import fit_supervised


from ..models.tlio_resnet import BasicBlock1D, ResNet1D


TLIO_SOURCE = "self-contained TLIO-compatible ResNet implementation; see NOTICE"

MIN_LOG_STD = math.log(1e-3)


def rotate_features_xy(features, angle):
    output = features.copy()
    output[:, :3] = common.rotate_xy(output[:, :3], angle)
    output[:, 3:6] = common.rotate_xy(output[:, 3:6], angle)
    return output


def rodrigues_rotate(values, axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        values * cosine
        + np.cross(axis[None], values) * sine
        + axis[None] * np.sum(values * axis[None], axis=1, keepdims=True) * (1.0 - cosine)
    )


class TLIODataset(Dataset):
    def __init__(self, sequences, window_size, stride, augment):
        self.sequences = sequences
        self.window_size = window_size
        self.augment = augment
        self.index = []
        for sequence_id, sequence in enumerate(sequences):
            for start in common.valid_window_starts(sequence, window_size, stride):
                self.index.append((sequence_id, int(start)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        sequence_id, start = self.index[item]
        sequence = self.sequences[sequence_id]
        end = start + self.window_size
        yaw = float(sequence.yaw_rel[start])
        features = rotate_features_xy(sequence.world_features[start:end], -yaw)
        target = common.rotate_xy(
            (sequence.position3[end] - sequence.position3[start])[None],
            -yaw,
        )[0]
        if self.augment:
            gyro_bias = np.random.uniform(-0.05, 0.05, size=(1, 3)).astype(np.float32)
            accel_bias = np.random.uniform(-0.2, 0.2, size=(1, 3)).astype(np.float32)
            features[:, :3] += gyro_bias
            features[:, 3:6] += accel_bias

            direction = np.random.uniform(-math.pi, math.pi)
            tilt = np.random.uniform(0.0, math.radians(5.0))
            axis = np.asarray([math.cos(direction), math.sin(direction), 0.0], dtype=np.float32)
            features[:, :3] = rodrigues_rotate(features[:, :3], axis, tilt)
            features[:, 3:6] = rodrigues_rotate(features[:, 3:6], axis, tilt)

            yaw_augmentation = np.random.uniform(-math.pi, math.pi)
            features = rotate_features_xy(features, yaw_augmentation)
            target = common.rotate_xy(target[None], yaw_augmentation)[0]
        return (
            torch.from_numpy(features.astype(np.float32).T),
            torch.from_numpy(target.astype(np.float32)),
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "all"), nargs="?", default="all")
    common.add_data_arguments(parser)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    common.finish_data_arguments(args)
    if args.output_root is None:
        args.output_root = (
            common.REPO_ROOT / "CoWear" / "results" / "classic_aipdr" / f"tlio_net_{args.role}"
        )
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(window_size):
    model = ResNet1D(BasicBlock1D, 6, 3, [2, 2, 2, 2], window_size // 32 + 1)
    with torch.no_grad():
        model(torch.zeros(2, 6, window_size))
    return model


def likelihood_loss(mean, logstd, target, epoch):
    logstd = torch.clamp(logstd, min=MIN_LOG_STD)
    if epoch < 10:
        logstd = logstd.detach()
    return torch.mean((mean - target) ** 2 / (2.0 * torch.exp(2.0 * logstd)) + logstd)


def _nll_step(model, loader, device, optimizer, epoch):
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        mean, logstd = model(features)
        loss = likelihood_loss(mean, logstd, target, epoch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(features)
        count += len(features)
    return total / max(count, 1)


def train(args, device):
    train_sequences = common.read_sequences(args, "train")
    val_sequences = common.read_sequences(args, "val")
    train_set = TLIODataset(
        train_sequences,
        args.window_size,
        args.stride,
        augment=not args.disable_augmentation,
    )
    val_set = TLIODataset(val_sequences, args.window_size, args.stride, augment=False)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.loader_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = build_model(args.window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **common.data_provenance(args),
        "method": "TLIO-Net",
        "official_source": str(TLIO_SOURCE),
        "architecture": "official 1D ResNet18 with displacement and diagonal log-standard-deviation heads",
        "estimator": "network-only overlapping displacement integration",
        "full_stochastic_cloning_ekf": False,
        "input": "gravity/world-aligned gyroscope + accelerometer in window-start yaw frame",
        "target": "3D displacement in window-start gravity-aligned frame",
        "window_size": args.window_size,
        "window_duration_s": args.window_size / args.sample_rate_hz,
        "stride": args.stride,
        "augmentation": not args.disable_augmentation,
        "gyro_bias_range": 0.05,
        "accel_bias_range": 0.2,
        "gravity_perturbation_deg": 5.0,
        "random_yaw": not args.disable_augmentation,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "seed": args.seed,
    }
    (args.output_root / "train_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    fit = fit_supervised(
        model, train_loader, val_loader, device, optimizer,
        _nll_step, _nll_step, args.epochs, args.patience, "tlio",
    )
    torch.save(
        {
            "model_state_dict": fit.best_state,
            "epoch": fit.best_epoch,
            "val_loss": fit.best_val_loss,
            "window_size": args.window_size,
            "stride": args.stride,
            "data_provenance": common.data_provenance(args),
        },
        checkpoint_dir / "best.pt",
    )
    (args.output_root / "history.json").write_text(json.dumps(fit.history, indent=2) + "\n", encoding="utf-8")
    print(f"[tlio] best_val={fit.best_val_loss:.6f}", flush=True)


def predict_displacements(model, sequence, starts, window_size, device):
    means, logstds = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), 1024):
            selected = starts[offset:offset + 1024]
            windows = []
            for start in selected:
                feature = rotate_features_xy(
                    sequence.world_features[start:start + window_size],
                    -float(sequence.yaw_rel[start]),
                )
                windows.append(feature.T)
            mean, logstd = model(torch.from_numpy(np.stack(windows)).to(device))
            means.append(mean.cpu().numpy())
            logstds.append(logstd.cpu().numpy())
    return np.concatenate(means), np.concatenate(logstds)


def validate_checkpoint(checkpoint, args):
    expected = common.data_provenance(args)
    actual = checkpoint.get("data_provenance", {})
    mismatch = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatch:
        raise ValueError(f"TLIO checkpoint provenance mismatch: {', '.join(mismatch)}")


def evaluate(args, device):
    checkpoint = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, args)
    model = build_model(checkpoint["window_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = []
    uncertainty_rows = []
    duration_s = checkpoint["window_size"] / args.sample_rate_hz
    step_s = checkpoint["stride"] / args.sample_rate_hz
    for sequence in common.read_sequences(args, args.eval_split):
        starts = common.valid_window_starts(sequence, checkpoint["window_size"], checkpoint["stride"])
        starts = common.longest_regular_run(starts, checkpoint["stride"])
        if not len(starts):
            continue
        displacement, logstd = predict_displacements(
            model,
            sequence,
            starts,
            checkpoint["window_size"],
            device,
        )
        for index, start in enumerate(starts):
            displacement[index] = common.rotate_xy(
                displacement[index:index + 1],
                float(sequence.yaw_rel[start]),
            )[0]
        velocity = displacement[:, :2] / duration_s
        prediction = np.empty_like(velocity)
        prediction[0] = sequence.position[starts[0]]
        if len(prediction) > 1:
            prediction[1:] = prediction[0] + np.cumsum(velocity[:-1] * step_s, axis=0)
        truth = sequence.position[starts]
        predictions.append((sequence.base_id, truth, prediction.astype(np.float32), starts))
        uncertainty_rows.append(
            {
                "base_id": sequence.base_id,
                "mean_sigma_x_m": float(np.mean(np.exp(logstd[:, 0]))),
                "mean_sigma_y_m": float(np.mean(np.exp(logstd[:, 1]))),
                "mean_sigma_z_m": float(np.mean(np.exp(logstd[:, 2]))),
            }
        )
    evaluation_root = args.evaluation_output_root or args.output_root
    common.write_evaluation(evaluation_root, "TLIO-Net", predictions)
    common.write_csv(evaluation_root / "uncertainty_summary.csv", uncertainty_rows)
    print(f"[tlio] evaluated {len(predictions)} {args.eval_split} trajectories", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[tlio] device={device} source={TLIO_SOURCE} loader={common.DATA_LOADER_VERSION}", flush=True)
    if args.command in {"prepare", "all"}:
        common.prepare_cache(args)
    if args.command in {"train", "all"}:
        train(args, device)
    if args.command in {"evaluate", "all"}:
        evaluate(args, device)


if __name__ == "__main__":
    main()
