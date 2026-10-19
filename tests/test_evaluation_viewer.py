from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apps.experiment_viewer.evaluation_figures import (
    build_evaluation_metric_figure,
    build_evaluation_sample_figure,
    build_segment_distribution_figure,
)
from apps.experiment_viewer.services import (
    EVALUATION_METRIC_BY_KEY,
    EvaluationStore,
)


class EvaluationViewerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary_directory.name) / "outputs"
        self.run_directory = self.output_root / "baselines" / "position_hold" / "run_1"
        self.run_directory.mkdir(parents=True)
        horizon_path = self.run_directory / "metrics_by_horizon.csv"
        horizon_path.write_text(
            "horizon_s,candidate_origins,evaluated_targets,coverage,horizontal_rmse_m\n"
            "1,100,90,0.9,12.5\n"
            "10,80,64,0.8,45.0\n",
            encoding="utf-8",
        )
        segment_path = self.run_directory / "metrics_by_segment.parquet"
        pq.write_table(
            pa.table(
                {
                    "horizon_s": [1, 1, 10, 10],
                    "horizontal_rmse_m": [10.0, 15.0, 40.0, 50.0],
                }
            ),
            segment_path,
        )
        manifest = {
            "artifact": "position_hold_baseline_evaluation",
            "configuration": {
                "version": "position_hold_v1",
                "subsets": [1],
                "split": "test",
                "prediction_horizons_seconds": [1, 10],
            },
            "outputs": {
                "metrics_by_horizon": str(horizon_path),
                "metrics_by_segment": str(segment_path),
            },
            "statistics": {"evaluated_segments": 2},
        }
        (self.run_directory / "manifest.yaml").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_store_discovers_and_reads_compatible_run(self) -> None:
        store = EvaluationStore(self.output_root)

        runs = store.runs()
        rows = store.horizon_metrics(runs[0].run_id)
        values = store.segment_metric_values(
            runs[0].run_id, "horizontal_rmse_m", 10
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].model_type, "position_hold")
        self.assertEqual(runs[0].horizons, (1, 10))
        self.assertEqual(rows[0]["candidate_origins"], 100)
        self.assertEqual(rows[1]["horizontal_rmse_m"], 45.0)
        self.assertEqual(values, (40.0, 50.0))

    def test_evaluation_figures_share_model_series(self) -> None:
        store = EvaluationStore(self.output_root)
        run = store.runs()[0]
        rows = store.horizon_metrics(run.run_id)
        metric = EVALUATION_METRIC_BY_KEY["horizontal_rmse_m"]

        error_figure = build_evaluation_metric_figure([(run.label, rows)], metric)
        sample_figure = build_evaluation_sample_figure([(run.label, rows)])
        distribution_figure = build_segment_distribution_figure(
            [(run.label, (40.0, 50.0))], metric, 10
        )

        self.assertEqual(tuple(error_figure.data[0].x), (1, 10))
        self.assertEqual(tuple(error_figure.data[0].y), (12.5, 45.0))
        self.assertEqual(len(sample_figure.data), 2)
        self.assertEqual(tuple(sample_figure.data[1].y), (90, 64))
        self.assertEqual(distribution_figure.data[0].type, "box")
        self.assertEqual(tuple(distribution_figure.data[0].y), (40.0, 50.0))


if __name__ == "__main__":
    unittest.main()
