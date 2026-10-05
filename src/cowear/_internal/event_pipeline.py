#!/usr/bin/env python3
"""Single-device variable-event odometry with gyro-relative pose residuals.

Each event uses the shared, jointly detected boundary and is a variable-length
sample.  The packed BiLSTM receives no event phase or duration feature.  It
predicts the event-start local displacement and a small correction to the
gyro-preintegrated relative rotation::

    delta_R_pred = delta_R_gyro @ Exp(residual_rotvec)

Only the first event receives an initial pose.  All later position and attitude
states are recursive predictions; no later ground-truth pose is injected.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .joint_sessions import build_sessions  # noqa: E402
from .. import INPUT_FEATURES


ROLES = ("mobile", "watch", "rokid")
# Keep the session-cache key stable: the already materialized session objects
# are portable.  Event tensors have their own version because v2 serialized a
# process-local dataclass.
CACHE_VERSION = "paper_v1_joint_vertical_gyro_boundaries_device6d_published_extrinsic_v2"
EVENT_CACHE_VERSION = "paper_v1_device6d_portable_items_v2"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Identify every preprocessing choice that changes cached tensors."""
    split_index = args.split_index.resolve()
    return {
        "version": CACHE_VERSION,
        "processed_root": str(args.processed_root.resolve()),
        "split_index": str(split_index),
        "split_index_mtime_ns": split_index.stat().st_mtime_ns,
        "calibration_mode": args.calibration_mode,
        "sample_rate_hz": args.sample_rate_hz,
        "max_gap_s": args.max_gap_s,
        "min_duration_s": args.min_duration_s,
        "smooth_s": args.smooth_s,
        "gravity_s": args.gravity_s,
        "truth_max_gap_s": args.truth_max_gap_s,
        "truth_max_jump_deg": args.truth_max_jump_deg,
        "truth_max_angular_rate_dps": args.truth_max_angular_rate_dps,
        "truth_max_speed_mps": args.truth_max_speed_mps,
        "imu_gap_floor_s": args.imu_gap_floor_s,
        "watch_imu_gap_floor_s": args.watch_imu_gap_floor_s,
        "imu_gap_median_factor": args.imu_gap_median_factor,
        "imu_gap_iqr_factor": args.imu_gap_iqr_factor,
        "imu_gap_ceiling_s": args.imu_gap_ceiling_s,
    }


def save_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cache_metadata_matches(
    cached: dict[str, object] | None, current: dict[str, object]
) -> bool:
    """Accept a cache after a path-only relocation of the same dataset.

    Dataset/result cleanup can rename the processed-data and split directories
    without changing any tensor-producing setting.  The split mtime and every
    preprocessing option must still match; only the two absolute paths may
    differ.
    """
    if cached == current:
        return True
    if not isinstance(cached, dict):
        return False
    path_keys = {"processed_root", "split_index"}
    return {
        key: value for key, value in cached.items() if key not in path_keys
    } == {
        key: value for key, value in current.items() if key not in path_keys
    }


def load_or_build_sessions(args: argparse.Namespace):
    metadata = cache_metadata(args)
    path = args.cache_dir / "sessions.pt"
    if path.is_file() and not args.rebuild_cache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if cache_metadata_matches(payload.get("metadata"), metadata):
            sessions = payload.get("sessions")
            if isinstance(sessions, list):
                print(f"session_cache=hit path={path} sessions={len(sessions)}", flush=True)
                return sessions
        print(f"session_cache=stale path={path}; rebuilding", flush=True)
    print(f"session_cache=miss path={path}; building sessions", flush=True)
    sessions = build_sessions(args)
    save_cache(path, {"metadata": metadata, "sessions": sessions})
    print(f"session_cache=written path={path} sessions={len(sessions)}", flush=True)
    return sessions


def rotvec_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) exponential map with a finite zero limit."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )
    theta2 = (vector * vector).sum(-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2 + 1e-8)
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(
        *vector.shape[:-1], 3, 3
    )
    return eye + a * skew + b * (skew @ skew)


def geodesic_angle(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


def integrate_gyro_local_to_world(session, role: str, fs: float) -> np.ndarray:
    """Integrate gyro once and express each sample in its gravity-local frame.

    The arbitrary initial device rotation cancels in event-relative deltas, so
    no GT attitude is used to construct the gyro baseline.
    """
    device_to_ref = np.eye(3, dtype=np.float32)
    output = np.empty((len(session.t), 3, 3), dtype=np.float32)
    for index in range(len(output)):
        output[index] = device_to_ref @ session.frames[role][index].T
        if index + 1 < len(output):
            device_to_ref = device_to_ref @ Rotation.from_rotvec(
                session.raw_features[role][index, :3] / fs
            ).as_matrix().astype(np.float32)
    return output


@dataclass
class ResidualEvent:
    x: np.ndarray
    local_disp: np.ndarray
    global_disp: np.ndarray
    start_rotation: np.ndarray
    gt_delta_rotation: np.ndarray
    gyro_delta_rotation: np.ndarray
    edge_valid: bool
    base_id: str
    a: int
    b: int


class ResidualEventDataset(Dataset):
    """One complete joint-boundary event per variable-length item."""

    def __init__(self, sessions, split: str, role: str, sample_rate_hz: float = 100.0):
        self.items: list[ResidualEvent] = []
        fs = float(sample_rate_hz)
        for session in sessions:
            if session.split != split:
                continue
            gyro_local_to_world = integrate_gyro_local_to_world(session, role, fs)
            gt_local_to_world = np.matmul(
                Rotation.from_quat(session.quat[role]).as_matrix().astype(np.float32),
                session.frames[role].transpose(0, 2, 1),
            )
            position = session.pos[role].astype(np.float32)
            # Event models consume only device-provided gyro and acceleration.
            features = session.raw_features[role][:, :INPUT_FEATURES].astype(np.float32)
            # The jointly detected boundaries establish a common temporal
            # segmentation across devices.  They are not model features:
            # packed sequences encode variable length without phase or a
            # duration scalar.
            for a_raw, b_raw in session.boundaries:
                a, b = int(a_raw), int(b_raw)
                if b <= a + 3:
                    continue
                # Both targets are relative quantities in the event-start
                # gravity frame.  The world-frame Vicon pose is used only to
                # construct training labels, never injected during rollout.
                global_disp = position[b] - position[a]
                start_rotation = gt_local_to_world[a]
                local_disp = start_rotation.T @ global_disp
                gt_delta = start_rotation.T @ gt_local_to_world[b]
                gyro_delta = gyro_local_to_world[a].T @ gyro_local_to_world[b]
                self.items.append(
                    ResidualEvent(
                        x=features[a : b + 1],
                        local_disp=local_disp.astype(np.float32),
                        global_disp=global_disp.astype(np.float32),
                        start_rotation=start_rotation.astype(np.float32),
                        gt_delta_rotation=gt_delta.astype(np.float32),
                        gyro_delta_rotation=gyro_delta.astype(np.float32),
                        edge_valid=bool(session.edge_valid[role][a:b].all()),
                        base_id=session.base_id,
                        a=a,
                        b=b,
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ResidualEvent:
        return self.items[index]


def load_or_build_event_dataset(
    args: argparse.Namespace, sessions, role: str, split: str
) -> ResidualEventDataset:
    metadata = {
        **cache_metadata(args),
        "event_cache_version": EVENT_CACHE_VERSION,
        "role": role,
        "split": split,
    }
    path = args.cache_dir / f"events_{role}_{split}.pt"
    if path.is_file() and not args.rebuild_cache:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cached_items = payload.get("items")
            if payload.get("metadata") == metadata and isinstance(cached_items, list):
                dataset = ResidualEventDataset.__new__(ResidualEventDataset)
                dataset.items = [ResidualEvent(**item) for item in cached_items]
                print(f"event_cache=hit role={role} split={split} events={len(dataset)}", flush=True)
                return dataset
            print(f"event_cache=stale role={role} split={split}; rebuilding", flush=True)
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError):
            # v2 used a __main__ dataclass pickle and cannot be reopened from
            # a later process.  Rebuild once from the portable session cache.
            print(f"event_cache=unreadable role={role} split={split}; rebuilding", flush=True)
    dataset = ResidualEventDataset(sessions, split, role, args.sample_rate_hz)
    save_cache(path, {"metadata": metadata, "items": [asdict(item) for item in dataset.items]})
    print(f"event_cache=written role={role} split={split} events={len(dataset)}", flush=True)
    return dataset


def collate_events(batch: Sequence[ResidualEvent]) -> dict[str, torch.Tensor | list]:
    lengths = torch.tensor([len(item.x) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    count = len(batch)
    x = torch.zeros(count, maximum, INPUT_FEATURES, dtype=torch.float32)
    for row, item in enumerate(batch):
        x[row, : len(item.x)] = torch.from_numpy(item.x)
    return {
        "x": x,
        "lengths": lengths,
        "local_disp": torch.from_numpy(np.stack([item.local_disp for item in batch])),
        "global_disp": torch.from_numpy(np.stack([item.global_disp for item in batch])),
        "start_rotation": torch.from_numpy(np.stack([item.start_rotation for item in batch])),
        "gt_delta_rotation": torch.from_numpy(np.stack([item.gt_delta_rotation for item in batch])),
        "gyro_delta_rotation": torch.from_numpy(np.stack([item.gyro_delta_rotation for item in batch])),
        "edge_valid": torch.tensor([item.edge_valid for item in batch], dtype=torch.bool),
        "base_id": [item.base_id for item in batch],
        "a": [item.a for item in batch],
        "b": [item.b for item in batch],
    }


class EventResidualMotion(nn.Module):
    """Packed BiLSTM with local displacement and relative gyro-residual heads."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden: int = 128,
        rotation_residual_limit_rad: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.input = nn.Sequential(nn.Linear(INPUT_FEATURES, hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.displacement = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.rotation = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.displacement[-1].weight)
        nn.init.zeros_(self.displacement[-1].bias)
        nn.init.zeros_(self.rotation[-1].weight)
        nn.init.zeros_(self.rotation[-1].bias)
        self.register_buffer(
            "rotation_limit",
            torch.full((3,), float(rotation_residual_limit_rad), dtype=torch.float32),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        gyro_delta_rotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        value = (x - self.feature_mean) / self.feature_std
        latent = self.input(value)
        packed = pack_padded_sequence(latent, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        latent, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        index = (lengths - 1).to(x.device)
        feature = latent[torch.arange(len(lengths), device=x.device), index]
        local_disp = self.displacement(feature)
        residual = torch.tanh(self.rotation(feature)) * self.rotation_limit
        # The residual is a relative rotation in the gyro-predicted event-end
        # frame.  Nothing here predicts an absolute world orientation.
        delta_rotation = gyro_delta_rotation @ rotvec_to_matrix(residual)
        return {
            "local_disp": local_disp,
            "rotation_residual": residual,
            "delta_rotation": delta_rotation,
        }


def event_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor | list]) -> tuple[torch.Tensor, dict[str, float]]:
    start_rotation = batch["start_rotation"]
    target_local = batch["local_disp"]
    target_global = batch["global_disp"]
    predicted_global = torch.einsum("bij,bj->bi", start_rotation, pred["local_disp"])
    local = F.smooth_l1_loss(pred["local_disp"], target_local, beta=0.10)
    global_loss = F.smooth_l1_loss(predicted_global, target_global, beta=0.10)
    attitude = geodesic_angle(pred["delta_rotation"], batch["gt_delta_rotation"]).mean()
    residual_penalty = pred["rotation_residual"].square().mean()
    loss = 0.5 * local + global_loss + 0.25 * attitude + 0.02 * residual_penalty
    return loss, {
        "local": float(local.detach()),
        "global": float(global_loss.detach()),
        "attitude_deg": float(torch.rad2deg(attitude).detach()),
        "residual_deg": float(torch.rad2deg(torch.linalg.vector_norm(pred["rotation_residual"], dim=-1).mean()).detach()),
    }


def run_epoch(model, loader, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "local": 0.0, "global": 0.0, "attitude_deg": 0.0, "residual_deg": 0.0}
    count = 0
    for batch in loader:
        tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            pred = model(tensor_batch["x"], tensor_batch["lengths"], tensor_batch["gyro_delta_rotation"])
            loss, terms = event_loss(pred, tensor_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        size = len(tensor_batch["lengths"])
        count += size
        totals["loss"] += float(loss.detach()) * size
        for key, value in terms.items():
            totals[key] += value * size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_events(model, dataset: ResidualEventDataset, device: torch.device, batch_size: int) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_events)
    model.eval()
    output: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for batch in loader:
        tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        pred = model(tensor_batch["x"], tensor_batch["lengths"], tensor_batch["gyro_delta_rotation"])
        for row, (base_id, a, b) in enumerate(zip(batch["base_id"], batch["a"], batch["b"])):
            output[(base_id, int(a), int(b))] = {
                "local_disp": pred["local_disp"][row].cpu().numpy(),
                "delta_rotation": pred["delta_rotation"][row].cpu().numpy(),
                "gyro_delta_rotation": batch["gyro_delta_rotation"][row].numpy(),
            }
    return output


def rollout_session(
    session,
    role: str,
    dataset: ResidualEventDataset,
    predictions: dict,
    sample_rate_hz: float,
    pose_mode: str = "residual",
    return_arrays: bool = False,
):
    items = sorted((item for item in dataset.items if item.base_id == session.base_id), key=lambda item: item.a)
    if not items:
        return None
    first = items[0].a
    current_rotation = items[0].start_rotation.astype(np.float64)
    predicted = [session.pos[role][first].astype(np.float64)]
    target = [session.pos[role][first].astype(np.float64)]
    indices = [first]
    cursor = first
    for item in items:
        if item.a != cursor:
            break
        value = predictions[(item.base_id, item.a, item.b)]
        predicted.append(predicted[-1] + current_rotation @ value["local_disp"])
        target.append(session.pos[role][item.b].astype(np.float64))
        indices.append(item.b)
        if pose_mode == "residual":
            delta_rotation = value["delta_rotation"]
        elif pose_mode == "gyro":
            delta_rotation = value["gyro_delta_rotation"]
        else:
            raise ValueError(f"unknown pose mode: {pose_mode}")
        current_rotation = current_rotation @ delta_rotation
        cursor = item.b
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    indices = np.asarray(indices, dtype=np.int64)
    valid = session.node_valid[role][indices]
    if len(predicted) < 2 or not np.any(valid):
        return None
    error = predicted - target
    horizontal = np.linalg.norm(error[:, [0, 2]], axis=1)
    error_3d = np.linalg.norm(error, axis=1)
    row = {
        "base_id": session.base_id,
        "events": int(len(predicted) - 1),
        "duration_s": float((indices[-1] - indices[0]) / float(sample_rate_hz)),
        "valid_node_fraction": float(valid.mean()),
        "ate_horizontal_m": float(horizontal[valid].mean()),
        "ate_3d_m": float(error_3d[valid].mean()),
        "endpoint_horizontal_m": float(horizontal[np.flatnonzero(valid)[-1]]),
        "endpoint_3d_m": float(error_3d[np.flatnonzero(valid)[-1]]),
    }
    if return_arrays:
        return row, {"predicted": predicted, "target": target, "indices": indices}
    return row


def evaluate(
    model,
    sessions,
    dataset,
    split: str,
    role: str,
    device: torch.device,
    batch_size: int,
    sample_rate_hz: float,
    pose_mode: str = "residual",
    return_details: bool = False,
):
    predictions = predict_events(model, dataset, device, batch_size)
    rows, arrays = [], {}
    for session in sessions:
        if session.split != split:
            continue
        result = rollout_session(
            session,
            role,
            dataset,
            predictions,
            sample_rate_hz,
            pose_mode,
            return_arrays=return_details,
        )
        if result is None:
            continue
        if return_details:
            row, values = result
            arrays[session.base_id] = values
        else:
            row = result
        rows.append(row)
    return rows, arrays


def summarize(rows: Sequence[dict]) -> dict:
    keys = ("ate_horizontal_m", "ate_3d_m", "endpoint_horizontal_m", "endpoint_3d_m")
    return {
        "sessions": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "median": float(np.median([row[key] for row in rows])),
                "p90": float(np.percentile([row[key] for row in rows], 90)),
            }
            for key in keys
        },
    }


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_role(args, sessions, role: str, device: torch.device) -> dict:
    seed_everything(args.seed)
    datasets = {
        split: load_or_build_event_dataset(args, sessions, role, split)
        for split in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError(f"empty event split for {role}: {[len(dataset) for dataset in datasets.values()]}")
    values = np.concatenate([item.x for item in datasets["train"].items], axis=0).astype(np.float64)
    mean = torch.tensor(values.mean(0), dtype=torch.float32)
    std = torch.tensor(np.sqrt(np.maximum(values.var(0), 1e-6)), dtype=torch.float32)
    model = EventResidualMotion(
        mean, std, args.hidden, args.rotation_residual_limit_rad
    ).to(device)
    initialized_keys = 0
    if args.initialize:
        checkpoint = torch.load(args.initialize, map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint.get("model_state_dict", {}))
        compatible = {
            key: value
            for key, value in source.items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        initialized_keys = len(compatible)
        print(f"role={role} initialized_shared_parameters={initialized_keys}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", collate_fn=collate_events, num_workers=args.num_workers)
        for split, dataset in datasets.items()
    }
    best = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    output_dir = args.output_dir / role
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, optimizer)
        val_rows, _ = evaluate(
            model, sessions, datasets["val"], "val", role, device,
            args.batch_size, args.sample_rate_hz,
        )
        val = summarize(val_rows)
        val_error = float(val["metrics"]["ate_horizontal_m"]["mean"])
        history.append({"epoch": epoch, "train": train, "val": val})
        print(f"role={role} epoch={epoch:03d} loss={train['loss']:.5f} val_ate={val_error:.4f}", flush=True)
        if val_error < best:
            best = val_error
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    checkpoint = {
        "model": model.state_dict(),
        "model_version": "cowear_single_device_event_gyro_residual_v2",
        "role": role,
        "architecture": "packed_bilstm_gyro_residual",
        "hidden": args.hidden,
        "feature_mean": mean,
        "feature_std": std,
        "rotation_baseline": "gyro event preintegration",
        "rotation_composition": "gyro_delta_rotation @ Exp(residual_rotvec)",
        "rotation_limit_rad": float(args.rotation_residual_limit_rad),
        "best_epoch": best_epoch,
        "calibration_mode": args.calibration_mode,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    test_rows, _ = evaluate(
        model, sessions, datasets["test"], "test", role, device,
        args.batch_size, args.sample_rate_hz,
    )
    gyro_rows, _ = evaluate(
        model, sessions, datasets["test"], "test", role, device,
        args.batch_size, args.sample_rate_hz, pose_mode="gyro",
    )
    val_rows, _ = evaluate(
        model, sessions, datasets["val"], "val", role, device,
        args.batch_size, args.sample_rate_hz,
    )
    write_rows(output_dir / "test_event_residual_rollout.csv", test_rows)
    write_rows(output_dir / "val_event_residual_rollout.csv", val_rows)
    metrics = {
        "meta": {
            "role": role,
            "train_sessions": len({item.base_id for item in datasets["train"].items}),
            "val_sessions": len({item.base_id for item in datasets["val"].items}),
            "test_sessions": len({item.base_id for item in datasets["test"].items}),
            "train_events": len(datasets["train"]),
            "val_events": len(datasets["val"]),
            "test_events": len(datasets["test"]),
            "event_protocol": "joint midpoint-to-midpoint event; one relative displacement and one relative pose residual per event",
            "event_input": "6D device-frame gyro+acceleration; no event phase or duration feature",
            "rotation_baseline": "gyro preintegration from event start to event end",
            "initial_alignment": "GT position and GT local-to-world pose at first event only",
            "calibration_mode": args.calibration_mode,
            "initialized_shared_parameters": initialized_keys,
        },
        "selection": {"best_epoch": best_epoch, "val_horizontal_ate_mean_m": best},
        "val": summarize(val_rows),
        "test": summarize(test_rows),
        "test_gyro_only": summarize(gyro_rows),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--split-index", type=Path, default=REPO_ROOT / "data" / "splits" / "session_split_seed2027.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "CoWear" / "results" / "ricloc_event_single_device_gyro_residual")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "CoWear" / "cache" / "event_relative_residual",
        help="Persistent cache for processed sessions and relative-motion event tensors.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached preprocessing and rebuild it from raw processed data.",
    )
    parser.add_argument("--roles", default="mobile,rokid,watch")
    parser.add_argument("--calibration-mode", choices=("published_extrinsic", "train_role_mean"), default="published_extrinsic")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--max-gap-s", type=float, default=0.35)
    parser.add_argument("--min-duration-s", type=float, default=4.0)
    parser.add_argument("--smooth-s", type=float, default=0.08)
    parser.add_argument("--gravity-s", type=float, default=0.55)
    parser.add_argument("--truth-max-gap-s", type=float, default=0.02)
    parser.add_argument("--truth-max-jump-deg", type=float, default=45.0)
    parser.add_argument("--truth-max-angular-rate-dps", type=float, default=2000.0)
    parser.add_argument("--truth-max-speed-mps", type=float, default=15.0)
    parser.add_argument("--imu-gap-floor-s", type=float, default=0.03)
    parser.add_argument("--watch-imu-gap-floor-s", type=float, default=0.10)
    parser.add_argument("--imu-gap-median-factor", type=float, default=2.5)
    parser.add_argument("--imu-gap-iqr-factor", type=float, default=3.0)
    parser.add_argument("--imu-gap-ceiling-s", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--rotation-residual-limit-rad",
        type=float,
        default=0.05,
        help="Per-event bound for each learned relative-rotation residual axis.",
    )
    parser.add_argument("--initialize", type=Path, default=None,
                        help="Optional event-model checkpoint; matching TCN/BiLSTM weights are reused.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rotation_residual_limit_rad <= 0.0:
        raise ValueError("--rotation-residual-limit-rad must be positive")
    roles = [value.strip() for value in args.roles.split(",") if value.strip()]
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles: {sorted(unknown)}")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sessions = load_or_build_sessions(args)
    summaries = {role: train_role(args, sessions, role, device) for role in roles}
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
