#!/usr/bin/env python3
"""RIC-Loc-compatible CoWear time and overlap helpers.

RIC-Loc's CoWear loaders put every stream on the same aligned absolute clock:

* Vicon truth uses ``groundtruth/align.csv:timestamp_ms / 1000``.
* IMU streams use ``alignedRelativeS + alignment_info.parameters.windowStartMs / 1000``.

For numerical stability, the manifest-first baselines store times relative to
the same session ``windowStartMs``.  This preserves the RIC-Loc clock while
avoiding interpolation on ~1.8e9-second epoch timestamps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


TIME_PROTOCOL_VERSION = "ricloc_absolute_clock_window_start_v1"


def session_dir_from_truth_path(truth_path: Path) -> Path:
    return truth_path.parent.parent


def window_start_s(session_dir: Path) -> float:
    info_path = session_dir / "alignment_info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        alignment_info = json.load(handle)
    return float(alignment_info["parameters"]["windowStartMs"]) / 1000.0


def truth_row_time_s(row: dict[str, str], session_dir: Path) -> float:
    return float(row["timestamp_ms"]) / 1000.0 - window_start_s(session_dir)


def imu_record_time_s(record: dict, session_dir: Path) -> float | None:
    """Return IMU time on the window-start-relative RIC-Loc clock."""
    if record.get("alignedRelativeS") is not None:
        return float(record["alignedRelativeS"])
    if record.get("alignedTimestampMs") is not None:
        return float(record["alignedTimestampMs"]) / 1000.0 - window_start_s(session_dir)
    return None


def continuous_intervals(timestamps: np.ndarray, max_gap_s: float) -> list[tuple[float, float]]:
    if len(timestamps) < 2:
        return []
    boundaries = np.flatnonzero(np.diff(timestamps) > max_gap_s) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries - 1, len(timestamps) - 1]
    return [
        (float(timestamps[start]), float(timestamps[end]))
        for start, end in zip(starts, ends)
    ]


def intersect_intervals(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            output.append((start, end))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return output


def longest_multistream_overlap(
    timestamp_streams: Iterable[np.ndarray],
    max_gap_s: float,
) -> tuple[float, float] | None:
    overlap: list[tuple[float, float]] | None = None
    for timestamps in timestamp_streams:
        intervals = continuous_intervals(timestamps, max_gap_s)
        if not intervals:
            return None
        overlap = intervals if overlap is None else intersect_intervals(overlap, intervals)
        if not overlap:
            return None
    return max(overlap, key=lambda interval: interval[1] - interval[0])
