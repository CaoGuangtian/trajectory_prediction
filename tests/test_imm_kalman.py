from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajectory_prediction.models.gated_motion_imm import (
    GatedMotionImmConfig,
    GatedMotionImmSource,
    evaluate_gated_motion_imm,
    predict_gated_motion_imm_batch,
)


class ImmKalmanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_path = self.root / "standardized.parquet"
        times = list(range(141))
        table = pa.table(
            {
                "aircraft": pa.array([7] * len(times), type=pa.int32()),
                "trajectory_id": pa.array([1] * len(times), type=pa.int64()),
                "segment_id": pa.array([10] * len(times), type=pa.int64()),
                "point_index": pa.array(times, type=pa.int32()),
                "timeAtServer": pa.array(times, type=pa.float64()),
                "longitude": pa.array(
                    [8.0 + value * 0.0001 for value in times],
                    type=pa.float64(),
                ),
                "latitude": pa.array(
                    [50.0 + value * 0.00005 for value in times],
                    type=pa.float64(),
                ),
                "geoAltitude": pa.array(
                    [1000.0 + value for value in times],
                    type=pa.float64(),
                ),
                "baroAltitude": pa.array(
                    [990.0 + value * 0.8 for value in times],
                    type=pa.float64(),
                ),
                "trusted": ["true"] * len(times),
                "split": ["test"] * len(times),
                "is_interpolated": [False] * len(times),
                "is_speed_suspicious": [False] * len(times),
            }
        )
        pq.write_table(table, self.source_path)
        self.config = GatedMotionImmConfig(
            version="gated_motion_imm_test_v1",
            implementation_version="gated_motion_imm_v1",
            model_type="gated_motion_imm",
            display_name="Gated Motion IMM",
            dataset_root=self.root,
            subsets=(1,),
            split="test",
            trusted_values=("true",),
            history_seconds=60,
            prediction_horizons_seconds=(1, 5, 10),
            origin_stride_seconds=1,
            minimum_history_points=10,
            minimum_history_span_seconds=30,
            skip_interpolated_history_updates=True,
            skip_suspicious_history_updates=True,
            exclude_interpolated_origins=True,
            exclude_suspicious_origins=True,
            exclude_interpolated_targets=True,
            exclude_suspicious_targets=True,
            measurement_std_e_m=30.0,
            measurement_std_n_m=30.0,
            measurement_std_u_m=30.0,
            initial_covariance_scale=10000.0,
            alpha_min=0.001,
            alpha_max=10.0,
            sigma_min=0.0001,
            sigma_max=200.0,
            process_q_min=0.05,
            process_q_max=400.0,
            ct_turn_rate_max=0.3,
            model_stickiness_min=0.90,
            model_stickiness_max=0.98,
            motion_gate_enabled=True,
            cv_fallback_history_points=31,
            maneuver_gate_block_points=11,
            maneuver_gate_min_speed_m_s=50.0,
            maneuver_gate_min_turn_rate_rad_s=0.002,
            maneuver_gate_min_acceleration_m_s2=0.3,
            maneuver_gate_consistency_ratio=0.5,
            prediction_batch_size=16,
            parquet_batch_rows=64,
            output_root=self.root / "outputs",
            source_path=self.root / "config.yaml",
            config_hash="testhash",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_straight_motion_is_causal_and_near_linear(self) -> None:
        time = np.arange(60, dtype=float) - 59.0
        history = np.column_stack([12.0 * time, 4.0 * time, 0.5 * time])
        batch = history[None, :, :]
        mask = np.ones((1, 60), dtype=bool)

        first = predict_gated_motion_imm_batch(batch, mask, (1, 5, 10), self.config)
        changed_past = batch.copy()
        changed_past[:, :10, 2] += 100.0
        second = predict_gated_motion_imm_batch(
            changed_past, mask, (1, 5, 10), self.config
        )

        self.assertEqual(first.positions_enu.shape, (1, 3, 3))
        self.assertFalse(bool(first.fallback_mask[0]))
        self.assertFalse(bool(first.maneuver_mask[0]))
        self.assertAlmostEqual(
            float(first.issue_model_probabilities[0].sum()), 1.0, places=10
        )
        np.testing.assert_allclose(
            first.issue_model_probabilities[0],
            np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            first.positions_enu[0, -1],
            np.array([120.0, 40.0, 5.0]),
            atol=3.0,
        )
        self.assertTrue(np.all(np.isfinite(second.positions_enu)))

    def test_persistent_turn_enables_ct_gate(self) -> None:
        time = np.arange(60, dtype=float) - 59.0
        speed = 180.0
        turn_rate = 0.01
        history = np.column_stack(
            [
                speed / turn_rate * np.sin(turn_rate * time),
                speed / turn_rate * (1.0 - np.cos(turn_rate * time)),
                np.zeros_like(time),
            ]
        )

        prediction = predict_gated_motion_imm_batch(
            history[None, :, :],
            np.ones((1, 60), dtype=bool),
            (1, 10),
            self.config,
        )

        self.assertTrue(bool(prediction.maneuver_mask[0]))
        self.assertTrue(bool(prediction.turn_maneuver_mask[0]))
        self.assertGreater(prediction.issue_model_probabilities[0, 1], 0.0)

    def test_persistent_acceleration_enables_singer_gate(self) -> None:
        time = np.arange(60, dtype=float) - 59.0
        history = np.column_stack(
            [
                150.0 * time + 0.5 * 0.6 * np.square(time),
                np.zeros_like(time),
                np.zeros_like(time),
            ]
        )

        prediction = predict_gated_motion_imm_batch(
            history[None, :, :],
            np.ones((1, 60), dtype=bool),
            (1, 10),
            self.config,
        )

        self.assertTrue(bool(prediction.maneuver_mask[0]))
        self.assertTrue(bool(prediction.acceleration_maneuver_mask[0]))
        self.assertGreater(prediction.issue_model_probabilities[0, 2], 0.0)

    def test_evaluation_writes_frontend_compatible_artifacts(self) -> None:
        output_directory = self.root / "result"
        manifest = evaluate_gated_motion_imm(
            [GatedMotionImmSource(1, self.source_path)],
            self.config,
            output_directory,
        )

        self.assertTrue((output_directory / "manifest.yaml").is_file())
        self.assertTrue((output_directory / "metrics_by_horizon.csv").is_file())
        self.assertTrue((output_directory / "metrics_by_segment.parquet").is_file())
        self.assertEqual(manifest["model_type"], "gated_motion_imm")
        self.assertEqual(manifest["statistics"]["unique_prediction_origins"], 72)
        self.assertEqual(manifest["statistics"]["maneuver_gated_origins"], 0)
        self.assertEqual(manifest["statistics"]["ols_cv_routed_origins"], 72)
        with (output_directory / "metrics_by_horizon.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["horizon_s"]) for row in rows], [1, 5, 10])
        self.assertLess(float(rows[-1]["horizontal_rmse_m"]), 10.0)
        self.assertEqual(rows[0]["barometric_vertical_mae_m"], "")
        persisted = json.loads(
            (output_directory / "manifest.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["configuration"]["history_points"], 60)

    def test_insufficient_history_falls_back_without_dropping_origins(self) -> None:
        fallback_config = replace(self.config, minimum_history_points=60)
        table = pq.read_table(self.source_path)
        interpolated = np.zeros(table.num_rows, dtype=bool)
        interpolated[::2] = True
        columns = {
            name: table[name]
            for name in table.column_names
            if name != "is_interpolated"
        }
        columns["is_interpolated"] = pa.array(interpolated)
        pq.write_table(pa.table(columns), self.source_path)

        manifest = evaluate_gated_motion_imm(
            [GatedMotionImmSource(1, self.source_path)],
            fallback_config,
            self.root / "fallback",
        )

        self.assertEqual(
            manifest["statistics"]["constant_velocity_fallback_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
