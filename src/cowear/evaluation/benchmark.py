"""Shared benchmark metrics and CSV result writing.

All baselines emit the same trajectory tuple format and use this module for
per-session metrics and CSV summaries.  Model-specific code should only
produce trajectories; it should not define another evaluation protocol.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..protocol.geometry import horizontal_ate


def trajectory_metrics(
    base_id: str, truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float | str]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] not in (2, 3):
        raise ValueError("truth and prediction must have matching shape [N, 2] or [N, 3]")
    if len(truth) == 0:
        raise ValueError("cannot evaluate an empty trajectory")
    error = prediction - truth
    truth_step = np.diff(truth, axis=0)
    pred_step = np.diff(prediction, axis=0)
    truth_path = float(np.sum(np.linalg.norm(truth_step, axis=1)))
    pred_path = float(np.sum(np.linalg.norm(pred_step, axis=1)))
    truth_speed = np.linalg.norm(truth_step, axis=1)
    pred_speed = np.linalg.norm(pred_step, axis=1)
    moving = (truth_speed > 0.005) & (pred_speed > 0.005)
    if np.any(moving):
        direction_cos = np.sum(truth_step[moving] * pred_step[moving], axis=1) / (
            truth_speed[moving] * pred_speed[moving]
        )
        direction_mean = float(np.mean(direction_cos))
    else:
        direction_mean = math.nan
    return {
        "base_id": base_id,
        "sessions": 1,
        "ate_rmse_m": float(horizontal_ate(prediction, truth)),
        "ate_mean_m": float(np.mean(np.linalg.norm(error, axis=1))),
        "endpoint_error_m": float(np.linalg.norm(error[-1])),
        "truth_path_m": truth_path,
        "pred_path_m": pred_path,
        "path_scale_ratio": pred_path / max(truth_path, 1e-8),
        "step_direction_cos_mean": direction_mean,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(output_root: Path, method: str, predictions: list[tuple]) -> list[dict]:
    """Write the common baseline trajectory schema.

    Each item is ``(base_id, truth, prediction)`` or
    ``(base_id, truth, prediction, sample_indices)``.
    """
    rows = []
    for item in predictions:
        if len(item) == 3:
            base_id, truth, prediction = item
            sample_indices = np.arange(len(truth), dtype=np.int64)
        elif len(item) == 4:
            base_id, truth, prediction, sample_indices = item
        else:
            raise ValueError(f"Unexpected trajectory tuple for {method}: {len(item)} entries")
        row = {"method": method, **trajectory_metrics(base_id, truth, prediction)}
        rows.append(row)
    if not rows:
        raise ValueError(f"No trajectories to evaluate for {method}")
    write_csv(output_root / "session_metrics.csv", rows)
    summary = [{
        "method": method,
        "sessions": len(rows),
        "ate_rmse_mean_m": f"{np.mean([row['ate_rmse_m'] for row in rows]):.6f}",
        "ate_rmse_median_m": f"{np.median([row['ate_rmse_m'] for row in rows]):.6f}",
        "endpoint_error_median_m": f"{np.median([row['endpoint_error_m'] for row in rows]):.6f}",
        "path_scale_ratio_median": f"{np.median([row['path_scale_ratio'] for row in rows]):.6f}",
        "step_direction_cos_mean_median": f"{np.median([row['step_direction_cos_mean'] for row in rows]):.6f}",
    }]
    write_csv(output_root / "summary.csv", summary)
    return rows
