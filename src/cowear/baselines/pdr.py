"""Public entry point for the paper's peak-trough PDR baseline."""

from ._pdr_support import (
    Calibration,
    Observation,
    SessionData,
    calibrate,
    detect_observation,
    evaluate,
    load_data,
    main,
    pdr_positions,
    read_specs,
    tune,
)

__all__ = [
    "Calibration", "Observation", "SessionData", "calibrate",
    "detect_observation", "evaluate", "load_data", "main",
    "pdr_positions", "read_specs", "tune",
]


if __name__ == "__main__":
    main()
