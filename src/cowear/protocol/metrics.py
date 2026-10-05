"""Paper metrics with stable, auditable CSV/JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import horizontal_ate


def summarize_ates(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty ATE list")
    array = np.asarray(values, dtype=np.float64)
    return {
        "sessions": int(array.size),
        "median_ate_m": float(np.median(array)),
        "mean_ate_m": float(np.mean(array)),
        "p25_ate_m": float(np.percentile(array, 25)),
        "p75_ate_m": float(np.percentile(array, 75)),
    }


def score_trajectories(trajectories: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    rows = []
    for session_id in sorted(trajectories):
        prediction, truth = trajectories[session_id]
        rows.append({"session_id": session_id, "ate_m": horizontal_ate(prediction, truth)})
    summary = summarize_ates([row["ate_m"] for row in rows])
    return {"summary": summary, "per_session": rows}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
