"""Small, dependency-light geometry primitives used by tests and evaluation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def world_to_local_displacement(rotation_local_to_world: np.ndarray, displacement_world: np.ndarray) -> np.ndarray:
    """Apply the paper's common-target world-to-observer-frame transform."""
    rotation = np.asarray(rotation_local_to_world, dtype=np.float64)
    displacement = np.asarray(displacement_world, dtype=np.float64)
    return np.einsum("...ji,...j->...i", rotation, displacement).astype(np.float32)


def local_to_world_displacement(rotation_local_to_world: np.ndarray, displacement_local: np.ndarray) -> np.ndarray:
    """Map a predicted observer-frame displacement into the world frame."""
    rotation = np.asarray(rotation_local_to_world, dtype=np.float64)
    displacement = np.asarray(displacement_local, dtype=np.float64)
    return np.einsum("...ij,...j->...i", rotation, displacement).astype(np.float32)


def propagate_gyro(rotation_local_to_world: np.ndarray, gyro_xyz: np.ndarray, sample_period_s: float) -> np.ndarray:
    """Propagate local-to-world orientation with the paper's gyro rule."""
    current = np.asarray(rotation_local_to_world, dtype=np.float64).copy()
    gyro = np.asarray(gyro_xyz, dtype=np.float64)
    if current.shape != (3, 3) or gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError("expected a (3, 3) initial rotation and (n, 3) gyro array")
    output = np.empty((len(gyro) + 1, 3, 3), dtype=np.float64)
    output[0] = current
    for index, sample in enumerate(gyro):
        current = current @ Rotation.from_rotvec(sample * sample_period_s).as_matrix()
        output[index + 1] = current
    return output.astype(np.float32)


def information_fusion(means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (n, 3)")
    if covariances.shape != (len(means), 3, 3):
        raise ValueError("covariances must have shape (n, 3, 3)")
    precision = np.linalg.inv(covariances + np.eye(3)[None] * 1e-6)
    covariance = np.linalg.inv(precision.sum(axis=0))
    mean = covariance @ np.sum(np.einsum("nij,nj->ni", precision, means), axis=0)
    return mean.astype(np.float32), covariance.astype(np.float32)


def horizontal_ate(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    count = min(len(prediction), len(truth))
    if count == 0:
        raise ValueError("cannot score an empty trajectory")
    error = prediction[:count] - truth[:count]
    axes = [0, 2] if error.shape[1] >= 3 else [0, 1]
    return float(np.sqrt(np.mean(np.sum(error[:, axes] ** 2, axis=1))))
