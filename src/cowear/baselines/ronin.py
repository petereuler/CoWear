#!/usr/bin/env python3
"""Train and evaluate a RoNIN ResNet on the canonical CoWear preprocessing."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .._internal import classic_data as common
from ..training import fit_supervised

RONIN_SOURCE = "https://github.com/nesl/RoNIN"


class RoNINDataset(Dataset):
    def __init__(self, sequences, window_size, stride, feature_mean, feature_std, target_mean, target_std, augment):
        self.sequences = sequences
        self.window_size = window_size
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.augment = augment
        self.index = []
        for sequence_id, sequence in enumerate(sequences):
            self.index.extend(
                (sequence_id, int(start))
                for start in common.valid_window_starts(sequence, window_size, stride)
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        sequence_id, start = self.index[item]
        sequence = self.sequences[sequence_id]
        end = start + self.window_size
        feature = sequence.world_features[start:end].copy()
        target = (
            (sequence.position[end] - sequence.position[start])
            / (sequence.dt_s * self.window_size)
        ).astype(np.float32)
        if self.augment:
            angle = random.uniform(-math.pi, math.pi)
            feature[:, :3] = common.rotate_xy(feature[:, :3], angle)
            feature[:, 3:6] = common.rotate_xy(feature[:, 3:6], angle)
            target = common.rotate_xy(target[None], angle)[0]
        feature = (feature - self.feature_mean) / self.feature_std
        target = (target - self.target_mean) / self.target_std
        return torch.from_numpy(feature.astype(np.float32).T), torch.from_numpy(target.astype(np.float32))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "all"), nargs="?", default="all")
    common.add_data_arguments(parser)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--disable-horizontal-augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    common.finish_data_arguments(args)
    if args.output_root is None:
        args.output_root = common.REPO_ROOT / "CoWear" / "results" / "classic_aipdr" / f"ronin_{args.role}"
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_statistics(sequences, window_size, stride):
    feature_sum = np.zeros(6, dtype=np.float64)
    feature_sq_sum = np.zeros(6, dtype=np.float64)
    feature_count = 0
    targets = []
    for sequence in sequences:
        starts = common.valid_window_starts(sequence, window_size, stride)
        if not len(starts):
            continue
        valid = np.zeros(len(sequence.world_features), dtype=bool)
        for start in starts:
            valid[start:start + window_size] = True
            end = start + window_size
            targets.append((sequence.position[end] - sequence.position[start]) / (sequence.dt_s * window_size))
        feature = sequence.world_features[valid].astype(np.float64)
        feature_sum += feature.sum(axis=0)
        feature_sq_sum += np.square(feature).sum(axis=0)
        feature_count += len(feature)
    feature_mean = feature_sum / feature_count
    feature_std = np.sqrt(np.maximum(feature_sq_sum / feature_count - feature_mean ** 2, 1e-8))
    targets = np.asarray(targets, dtype=np.float64)
    target_mean = targets.mean(axis=0)
    target_std = np.maximum(targets.std(axis=0), 1e-4)
    return feature_mean.astype(np.float32), feature_std.astype(np.float32), target_mean.astype(np.float32), target_std.astype(np.float32)


def make_dataset(sequences, args, stats, augment):
    return RoNINDataset(sequences, args.window_size, args.stride, *stats, augment)


def _mse_step(model, loader, device, optimizer, _epoch):
    total = count = 0
    for feature, target in loader:
        feature, target = feature.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(feature) - target) ** 2)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(feature)
        count += len(feature)
    return total / max(count, 1)


def train(args, device):
    train_sequences = common.read_sequences(args, "train")
    val_sequences = common.read_sequences(args, "val")
    stats = training_statistics(train_sequences, args.window_size, args.stride)
    train_set = make_dataset(train_sequences, args, stats, not args.disable_horizontal_augmentation)
    val_set = make_dataset(val_sequences, args, stats, False)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.loader_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = common.ronin.build_model(args.window_size, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **common.data_provenance(args),
        "method": "RoNIN-ResNet",
        "official_source": RONIN_SOURCE,
        "architecture": "official RoNIN 1D ResNet18",
        "input": "gyro-propagated world-frame gyroscope and accelerometer",
        "target": "2D average velocity in the benchmark frame",
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "horizontal_augmentation": not args.disable_horizontal_augmentation,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "seed": args.seed,
    }
    (args.output_root / "train_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    fit = fit_supervised(
        model, train_loader, val_loader, device, optimizer,
        _mse_step, _mse_step, args.epochs, args.patience, "ronin",
    )
    torch.save({
        "model_state_dict": fit.best_state,
        "epoch": fit.best_epoch,
        "val_loss": fit.best_val_loss,
        "window_size": args.window_size,
        "stride": args.stride,
        "dropout": args.dropout,
        "feature_mean": stats[0], "feature_std": stats[1],
        "target_mean": stats[2], "target_std": stats[3],
        "data_provenance": common.data_provenance(args),
    }, checkpoint_dir / "best.pt")
    history = [
        {"epoch": row["epoch"], "train_mse": row["train_loss"], "val_mse": row["val_loss"]}
        for row in fit.history
    ]
    (args.output_root / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"[ronin] best_val={fit.best_val_loss:.6f}", flush=True)


def evaluate(args, device):
    checkpoint = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    expected = common.data_provenance(args)
    mismatch = [key for key, value in expected.items() if checkpoint["data_provenance"].get(key) != value]
    if mismatch:
        raise ValueError(f"RoNIN checkpoint provenance mismatch: {', '.join(mismatch)}")
    model = common.ronin.build_model(checkpoint["window_size"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = []
    for sequence in common.read_sequences(args, args.eval_split):
        starts = common.longest_regular_run(
            common.valid_window_starts(sequence, checkpoint["window_size"], checkpoint["stride"]),
            checkpoint["stride"],
        )
        if not len(starts):
            continue
        output = []
        with torch.inference_mode():
            for offset in range(0, len(starts), 1024):
                ids = starts[offset:offset + 1024]
                windows = np.stack([
                    ((sequence.world_features[start:start + checkpoint["window_size"]] - checkpoint["feature_mean"]) / checkpoint["feature_std"]).T
                    for start in ids
                ]).astype(np.float32)
                output.append(model(torch.from_numpy(windows).to(device)).cpu().numpy())
        velocity = np.concatenate(output) * checkpoint["target_std"] + checkpoint["target_mean"]
        prediction = np.empty_like(velocity)
        prediction[0] = sequence.position[starts[0]]
        if len(prediction) > 1:
            prediction[1:] = prediction[0] + np.cumsum(
                velocity[:-1] * sequence.dt_s * checkpoint["stride"], axis=0
            )
        truth = sequence.position[starts]
        predictions.append((sequence.base_id, truth, prediction.astype(np.float32), starts))
    evaluation_root = args.evaluation_output_root or args.output_root
    common.write_evaluation(evaluation_root, "RoNIN-ResNet", predictions)
    print(f"[ronin] evaluated {len(predictions)} {args.eval_split} trajectories", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[ronin] device={device} loader={common.DATA_LOADER_VERSION}", flush=True)
    if args.command in {"prepare", "all"}:
        common.prepare_cache(args)
    if args.command in {"train", "all"}:
        train(args, device)
    if args.command in {"evaluate", "all"}:
        evaluate(args, device)


if __name__ == "__main__":
    main()
