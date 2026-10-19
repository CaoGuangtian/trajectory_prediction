from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajectory_prediction.baselines.constant_velocity import (
    ConstantVelocityConfig,
    ConstantVelocitySource,
    evaluate_constant_velocity,
)


class ConstantVelocityBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_path = self.root / "standardized.parquet"
        times = list(range(131))
        table = pa.table(
            {
                "aircraft": pa.array([7] * len(times), type=pa.int32()),
                "trajectory_id": pa.array([1] * len(times), type=pa.int64()),
                "segment_id": pa.array([10] * len(times), type=pa.int64()),
                "point_index": pa.array(times, type=pa.int32()),
                "timeAtServer": pa.array(times, type=pa.float64()),
                "longitude": pa.array(
                    [8.0 + value * 0.0001 for value in times], type=pa.float64()
                ),
                "latitude": pa.array(
                    [50.0 + value * 0.00005 for value in times], type=pa.float64()
                ),
                "geoAltitude": pa.array(
                    [1000.0 + value for value in times], type=pa.float64()
                ),
                "baroAltitude": pa.array(
                    [990.0 + value * 0.8 for value in times], type=pa.float64()
                ),
                "trusted": ["true"] * len(times),
                "split": ["test"] * len(times),
                "is_interpolated": [False] * len(times),
                "is_speed_suspicious": [False] * len(times),
            }
        )
        pq.write_table(table, self.source_path)
        self.config = ConstantVelocityConfig(
            version="constant_velocity_test_v1",
            dataset_root=self.root,
            subsets=(1,),
            split="test",
            history_seconds=30,
            prediction_horizons_seconds=(1, 5),
            origin_stride_seconds=1,
            exclude_interpolated_history=True,
            exclude_suspicious_history=True,
            minimum_history_points=5,
            minimum_history_span_seconds=10,
            exclude_interpolated_origins=True,
            exclude_suspicious_origins=True,
            exclude_interpolated_targets=True,
            exclude_suspicious_targets=True,
            output_root=self.root / "outputs",
            duckdb_memory_limit="256MB",
            duckdb_threads=1,
            source_path=self.root / "config.yaml",
            config_hash="testhash",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_straight_constant_velocity_has_near_zero_error(self) -> None:
        output_directory = self.root / "result"
        manifest = evaluate_constant_velocity(
            [ConstantVelocitySource(1, self.source_path)],
            self.config,
            output_directory,
        )

        with (output_directory / "metrics_by_horizon.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual([int(row["horizon_s"]) for row in rows], [1, 5])
        self.assertLess(float(rows[0]["horizontal_rmse_m"]), 0.001)
        self.assertLess(float(rows[1]["horizontal_rmse_m"]), 0.001)
        self.assertLess(float(rows[0]["vertical_rmse_m"]), 1e-8)
        self.assertEqual(manifest["statistics"]["velocity_fallback_origins"], 0)
        self.assertTrue((output_directory / "metrics_by_segment.parquet").is_file())
        persisted = json.loads(
            (output_directory / "manifest.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["model_type"], "constant_velocity")

    def test_insufficient_history_falls_back_without_dropping_origins(self) -> None:
        fallback_config = replace(self.config, minimum_history_points=1000)
        output_directory = self.root / "fallback"

        manifest = evaluate_constant_velocity(
            [ConstantVelocitySource(1, self.source_path)],
            fallback_config,
            output_directory,
        )

        self.assertEqual(manifest["statistics"]["velocity_fallback_rate"], 1.0)
        with (output_directory / "metrics_by_horizon.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(int(rows[0]["candidate_origins"]), 100)
        self.assertGreater(float(rows[0]["horizontal_rmse_m"]), 1.0)


if __name__ == "__main__":
    unittest.main()
