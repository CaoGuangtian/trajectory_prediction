from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajectory_prediction.baselines.position_hold import (
    PositionHoldConfig,
    PositionHoldSource,
    evaluate_position_hold,
)


class PositionHoldBaselineTests(unittest.TestCase):
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
                    [value * 0.0001 for value in times], type=pa.float64()
                ),
                "latitude": pa.array([0.0] * len(times), type=pa.float64()),
                "geoAltitude": pa.array(
                    [1000.0 + value for value in times], type=pa.float64()
                ),
                "baroAltitude": pa.array(
                    [990.0 + value for value in times], type=pa.float64()
                ),
                "trusted": ["true"] * len(times),
                "split": ["test"] * len(times),
                "is_interpolated": [value == 35 for value in times],
                "is_speed_suspicious": [value == 40 for value in times],
            }
        )
        pq.write_table(table, self.source_path)
        self.config = PositionHoldConfig(
            version="test_v1",
            dataset_root=self.root,
            subsets=(1,),
            split="test",
            required_history_points=60,
            prediction_horizons_seconds=(1, 5),
            origin_stride_seconds=1,
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

    def test_evaluation_writes_expected_aggregate_artifacts(self) -> None:
        output_directory = self.root / "result"
        manifest = evaluate_position_hold(
            [PositionHoldSource(1, self.source_path)],
            self.config,
            output_directory,
        )

        self.assertTrue((output_directory / "manifest.yaml").is_file())
        self.assertTrue((output_directory / "metrics_by_horizon.csv").is_file())
        self.assertTrue((output_directory / "metrics_by_segment.parquet").is_file())
        with (output_directory / "metrics_by_horizon.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["horizon_s"]) for row in rows], [1, 5])
        self.assertEqual(int(rows[0]["candidate_origins"]), 67)
        self.assertEqual(int(rows[0]["evaluated_targets"]), 67)
        self.assertEqual(int(rows[1]["candidate_origins"]), 67)
        self.assertEqual(int(rows[1]["evaluated_targets"]), 67)
        self.assertGreater(float(rows[0]["horizontal_mean_m"]), 0.0)
        self.assertGreater(float(rows[0]["vertical_mae_m"]), 0.0)

        segment_table = pq.read_table(output_directory / "metrics_by_segment.parquet")
        self.assertEqual(segment_table.num_rows, 2)
        self.assertEqual(segment_table["subset_id"].to_pylist(), [1, 1])
        self.assertEqual(segment_table["segment_id"].to_pylist(), [10, 10])
        self.assertEqual(manifest["statistics"]["evaluated_segments"], 1)

        persisted_manifest = json.loads(
            (output_directory / "manifest.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            persisted_manifest["artifact"],
            "position_hold_baseline_evaluation",
        )
        self.assertEqual(persisted_manifest["model_type"], "position_hold")
        self.assertEqual(
            persisted_manifest["evaluation_protocol"],
            "complete-window causal open-loop",
        )

    def test_multiple_subsets_are_combined_in_one_segment_file(self) -> None:
        second_path = self.root / "standardized_2.parquet"
        pq.write_table(pq.read_table(self.source_path), second_path)
        output_directory = self.root / "combined"

        evaluate_position_hold(
            [
                PositionHoldSource(1, self.source_path),
                PositionHoldSource(2, second_path),
            ],
            self.config,
            output_directory,
        )

        table = pq.read_table(output_directory / "metrics_by_segment.parquet")
        self.assertEqual(set(table["subset_id"].to_pylist()), {1, 2})


if __name__ == "__main__":
    unittest.main()
