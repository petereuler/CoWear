from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from cowear import INPUT_FEATURES, PAPER_ROLES
from cowear.protocol.config import PaperConfig, load_config
from cowear.protocol.data import paper_role, read_manifest, storage_role, validate_paper_split
from cowear.protocol.geometry import (
    horizontal_ate,
    information_fusion,
    local_to_world_displacement,
    propagate_gyro,
    world_to_local_displacement,
)
from cowear.protocol.checkpoints import read_checkpoint
from cowear.models.cowear_lstm import CoWearLSTM
from cowear.evaluation.paper import released_external_metrics, trajectory_ate
from cowear.baselines import BASELINES, get_baseline


class TestContracts(unittest.TestCase):
    def test_information_fusion_matches_identical_measurement(self) -> None:
        means = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        covariance = np.stack([np.eye(3), np.eye(3)])
        mean, fused_covariance = information_fusion(means, covariance)
        np.testing.assert_allclose(mean, means[0], atol=1e-6)
        np.testing.assert_allclose(fused_covariance, np.eye(3) * 0.5, atol=1e-5)

    def test_common_target_displacement_round_trip(self) -> None:
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        world = np.asarray([2.0, 3.0, 4.0])
        local = world_to_local_displacement(rotation, world)
        np.testing.assert_allclose(local_to_world_displacement(rotation, local), world, atol=1e-6)

    def test_gyro_propagation_uses_local_to_world_composition(self) -> None:
        rotations = propagate_gyro(np.eye(3), np.asarray([[0.0, 0.0, 2.0 * np.pi]]), 0.5)
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(rotations[1], np.diag([-1.0, -1.0, 1.0]), atol=1e-6)

    def test_horizontal_ate_uses_xz_for_three_dimensional_trajectories(self) -> None:
        truth = np.zeros((2, 3))
        prediction = np.asarray([[0.0, 100.0, 0.0], [3.0, -100.0, 4.0]])
        self.assertAlmostEqual(horizontal_ate(prediction, truth), np.sqrt(12.5))

    def test_external_trajectory_metric_uses_horizontal_xz(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(path, truth=np.zeros((2, 3)), prediction=np.asarray([[0.0, 100.0, 0.0], [3.0, -100.0, 4.0]]))
            score, points = trajectory_ate(path)
            self.assertEqual(points, 2)
            self.assertAlmostEqual(score, np.sqrt(12.5))

    def test_released_external_metrics_require_complete_unique_sessions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "per_session_ate.csv"
            fields = [
                "task", "method", "target", "estimator", "base_id", "ate_m",
                "trajectory_points", "segments",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for method in ("PDR", "RIDI", "RoNIN", "TLIO"):
                    for target in ("Phone", "Watch", "Glasses"):
                        for index in range(115):
                            writer.writerow({
                                "task": "task1_self", "method": method,
                                "target": target, "estimator": target,
                                "base_id": f"test/{index}", "ate_m": "1.0",
                                "trajectory_points": "2", "segments": "1",
                            })
                    for index in range(115):
                        writer.writerow({
                            "task": "benchmark_fusion", "method": method,
                            "target": "Phone", "estimator": "self_checkpoint_uniform",
                            "base_id": f"test/{index}", "ate_m": "1.0",
                            "trajectory_points": "2", "segments": "1",
                        })
            self.assertEqual(len(released_external_metrics(path)), 16 * 115)


    def test_manifest_split_is_frozen(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["base_id", "split", "available_roles"])
                writer.writeheader()
                for split, count in (("train", 217), ("val", 31), ("test", 115)):
                    for index in range(count):
                        writer.writerow({
                            "base_id": f"{split}/{index}",
                            "split": split,
                            "available_roles": "mobile,watch,rokid",
                        })
            self.assertEqual(validate_paper_split(path), {"train": 217, "val": 31, "test": 115})
            self.assertEqual(len(read_manifest(path)), 363)

    def test_public_manifest_has_disjoint_session_splits(self) -> None:
        path = Path(__file__).resolve().parents[1] / "splits" / "paper_v1_session_split.csv"
        rows = read_manifest(path)
        by_split = {
            split: {row.base_id for row in rows if row.split == split}
            for split in ("train", "val", "test")
        }
        self.assertEqual({len(values) for values in by_split.values()}, {217, 31, 115})
        self.assertTrue(by_split["train"].isdisjoint(by_split["val"]))
        self.assertTrue(by_split["train"].isdisjoint(by_split["test"]))
        self.assertTrue(by_split["val"].isdisjoint(by_split["test"]))

    def test_protocol_rejects_wrong_session_count(self) -> None:
        with self.assertRaises(ValueError):
            PaperConfig(split_counts=(1, 1, 1)).validate()

    def test_paper_protocol_uses_device_six_channel_input(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "paper_v1.toml")
        self.assertEqual(config.roles, PAPER_ROLES)
        self.assertEqual(config.input_features, INPUT_FEATURES)
        self.assertEqual(config.target_role, "phone")
        self.assertEqual(paper_role("mobile"), "phone")
        self.assertEqual(storage_role("glasses"), "rokid")
        model = CoWearLSTM(torch.zeros(INPUT_FEATURES), torch.ones(INPUT_FEATURES), 128)
        output = model(
            torch.zeros(2, 4, INPUT_FEATURES),
            torch.tensor([4, 3]),
            torch.eye(3).repeat(2, 1, 1),
        )
        self.assertEqual(tuple(output["local_mean"].shape), (2, 3))

    def test_checkpoint_metadata_is_required(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_checkpoint_metadata_accepts_canonical_paper_contract(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "good.pt"
            torch.save(
                {
                    "state_dict": {},
                    "metadata": {
                        "model_version": "cowear_lstm_native6d_v1",
                        "role": "phone",
                        "target": "phone",
                        "feature_frame": "device",
                        "split": "train_val_test_session_split",
                        "seed": 2027,
                        "input_features": INPUT_FEATURES,
                        "dataset_version": "CoWear-3device-363-v1",
                        "protocol_version": "paper-v1",
                    },
                },
                path,
            )
            self.assertEqual(read_checkpoint(path)["metadata"].input_features, INPUT_FEATURES)

    def test_checkpoint_metadata_rejects_wrong_feature_frame(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-frame.pt"
            torch.save(
                {
                    "metadata": {
                        "model_version": "cowear_lstm_native6d_v1",
                        "role": "phone",
                        "target": "phone",
                        "feature_frame": "gravity",
                        "split": "train_val_test_session_split",
                        "seed": 2027,
                        "input_features": INPUT_FEATURES,
                    }
                },
                path,
            )
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_public_baseline_registry_contains_real_modules(self) -> None:
        self.assertEqual(set(BASELINES), {"pdr", "ridi", "ronin", "tlio", "cowear_lstm"})
        self.assertTrue(all(spec.module.startswith("cowear.") for spec in BASELINES.values()))
        self.assertEqual(get_baseline("ronin").name, "RoNIN")
        with self.assertRaises(ValueError):
            get_baseline("not-a-paper-method")


if __name__ == "__main__":
    unittest.main()
