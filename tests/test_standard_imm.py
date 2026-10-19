from __future__ import annotations

import csv
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

from trajectory_prediction.models.standard_imm import (
    StandardImmConfig,
    StandardImmSource,
    evaluate_standard_imm,
    predict_standard_imm_batch,
)


class StandardImmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = StandardImmConfig(
            version="standard_imm_test_v1",
            implementation_version="standard_imm_v1",
            model_type="standard_imm",
            display_name="Standard IMM",
            dataset_root=self.root,
            subsets=(1,),
            split="test",
            trusted_values=(),
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
            model_names=("cv", "ct_left", "ct_right", "singer"),
            initial_model_probabilities=(0.7, 0.1, 0.1, 0.1),
            model_transition_matrix=(
                (0.97, 0.01, 0.01, 0.01),
                (0.01, 0.97, 0.01, 0.01),
                (0.01, 0.01, 0.97, 0.01),
                (0.01, 0.01, 0.01, 0.97),
            ),
            cv_process_q_xyz=(1.0, 1.0, 0.25),
            ct_process_q_xyz=(2.0, 2.0, 0.5),
            ct_turn_rate_rad_s=np.deg2rad(0.5),
            singer_alpha_xyz=(0.1, 0.1, 0.1),
            singer_sigma_xyz=(1.0, 1.0, 0.5),
            prediction_batch_size=16,
            parquet_batch_rows=64,
            output_root=self.root / "outputs",
            source_path=self.root / "config.yaml",
            config_hash="testhash",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_standard_imm_is_finite_and_probabilities_sum_to_one(self) -> None:
        time = np.arange(60, dtype=float) - 59.0
        history = np.column_stack([12.0 * time, 4.0 * time, 0.5 * time])
        result = predict_standard_imm_batch(
            history[None, :, :],
            np.ones((1, 60), dtype=bool),
            (1, 5, 10),
            self.config,
        )
        self.assertEqual(result.positions_enu.shape, (1, 3, 3))
        self.assertFalse(bool(result.fallback_mask[0]))
        self.assertTrue(np.all(np.isfinite(result.positions_enu)))
        self.assertAlmostEqual(float(result.issue_model_probabilities[0].sum()), 1.0, places=10)

    def test_single_cv_bank_is_a_cv_kalman_filter(self) -> None:
        config = replace(
            self.config,
            model_type="cv_kf",
            display_name="CV Kalman Filter",
            model_names=("cv",),
            initial_model_probabilities=(1.0,),
            model_transition_matrix=((1.0,),),
        )
        time = np.arange(60, dtype=float) - 59.0
        history = np.column_stack([12.0 * time, 4.0 * time, 0.5 * time])
        result = predict_standard_imm_batch(
            history[None, :, :], np.ones((1, 60), dtype=bool), (10,), config
        )
        np.testing.assert_allclose(result.issue_model_probabilities, [[1.0]])
        np.testing.assert_allclose(result.positions_enu[0, 0], [120.0, 40.0, 5.0], atol=5.0)

    def test_evaluation_writes_frontend_compatible_artifacts(self) -> None:
        source_path = self.root / "standardized.parquet"
        times = np.arange(141)
        pq.write_table(
            pa.table(
                {
                    "aircraft": pa.array([7] * times.size, type=pa.int32()),
                    "trajectory_id": pa.array([1] * times.size, type=pa.int64()),
                    "segment_id": pa.array([10] * times.size, type=pa.int64()),
                    "point_index": pa.array(times, type=pa.int32()),
                    "timeAtServer": pa.array(times, type=pa.float64()),
                    "longitude": pa.array(8.0 + times * 0.0001, type=pa.float64()),
                    "latitude": pa.array(50.0 + times * 0.00005, type=pa.float64()),
                    "geoAltitude": pa.array(1000.0 + times, type=pa.float64()),
                    "baroAltitude": pa.array(990.0 + times * 0.8, type=pa.float64()),
                    "trusted": ["true"] * times.size,
                    "split": ["test"] * times.size,
                    "is_interpolated": [False] * times.size,
                    "is_speed_suspicious": [False] * times.size,
                }
            ),
            source_path,
        )
        output = self.root / "result"
        manifest = evaluate_standard_imm(
            [StandardImmSource(1, source_path)], self.config, output
        )
        self.assertEqual(manifest["model_type"], "standard_imm")
        self.assertEqual(manifest["statistics"]["unique_prediction_origins"], 72)
        self.assertNotIn("ols_cv_routed_origins", manifest["statistics"])
        self.assertTrue((output / "metrics_by_segment.parquet").is_file())
        with (output / "metrics_by_horizon.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["horizon_s"]) for row in rows], [1, 5, 10])


if __name__ == "__main__":
    unittest.main()
