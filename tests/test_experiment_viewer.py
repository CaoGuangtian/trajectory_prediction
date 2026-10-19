from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apps.experiment_viewer.figures import (
    build_sampling_distribution_figure,
    build_sampling_interval_figure,
    build_spatial_figure,
    build_time_series_figure,
)
from apps.experiment_viewer.services import ProcessedTrajectoryStore, TrajectoryStore


class ExperimentViewerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temporary_directory.name) / "tracks.parquet"
        table = pa.table(
            {
                "aircraft": [7, 7, 7, 7, 12],
                "timeAtServer": [0.0, 1.0, 2.0, 20.0, 0.0],
                "longitude": [113.0, 113.1, 113.2, 113.3, 114.0],
                "latitude": [34.0, 34.1, 34.2, 34.3, 35.0],
                "geoAltitude": [1000.0, 1010.0, 1020.0, None, 800.0],
                "baroAltitude": [990.0, 1002.0, 1010.0, 1015.0, 790.0],
            }
        )
        pq.write_table(table, self.parquet_path)
        self.processed_path = (
            Path(self.temporary_directory.name) / "tracks_standardized_1hz.parquet"
        )
        processed_table = pa.table(
            {
                "aircraft": pa.array([7, 7, 7], type=pa.int32()),
                "trajectory_id": pa.array([1, 1, 1], type=pa.int64()),
                "segment_id": pa.array([10, 10, 11], type=pa.int64()),
                "point_index": pa.array([0, 1, 0], type=pa.int32()),
                "timeAtServer": [0.0, 1.0, 2.0],
                "longitude": [113.0, 113.1, 113.2],
                "latitude": [34.0, 34.1, 34.2],
                "geoAltitude": [1000.0, 1010.0, 1020.0],
                "baroAltitude": [990.0, 1000.0, 1010.0],
                "trusted": ["true", "true", "true"],
                "split": ["train", "train", "train"],
                "is_interpolated": [False, True, False],
                "is_speed_suspicious": [False, False, True],
            }
        )
        pq.write_table(processed_table, self.processed_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_store_summarizes_and_loads_track(self) -> None:
        store = TrajectoryStore(self.parquet_path)

        summaries = store.aircraft_summaries()
        track = store.load_track(7)

        self.assertEqual([summary.aircraft for summary in summaries], [7, 12])
        self.assertEqual(summaries[0].point_count, 4)
        self.assertEqual(track.point_count, 4)
        self.assertEqual(track.time, (0.0, 1.0, 2.0, 20.0))
        self.assertEqual(track.between(0.5, 2.0).time, (1.0, 2.0))

    def test_global_sampling_intervals_do_not_cross_aircraft(self) -> None:
        store = TrajectoryStore(self.parquet_path)

        distribution = store.global_sampling_distribution()

        self.assertEqual(distribution.total_count, 3)
        self.assertEqual(distribution.detail_count, 2)
        self.assertEqual(sum(distribution.detail_counts), 2)
        self.assertEqual(sum(distribution.gap_counts), 1)

    def test_processed_store_preserves_segment_and_quality_fields(self) -> None:
        store = ProcessedTrajectoryStore(self.processed_path)

        summary = store.aircraft_summaries()[0]
        track = store.load_track(7)

        self.assertEqual(summary.point_count, 3)
        self.assertEqual(summary.interpolated_point_count, 1)
        self.assertEqual(summary.suspicious_point_count, 1)
        self.assertEqual(summary.segment_count, 2)
        self.assertEqual(track.segment_id, (10, 10, 11))
        self.assertEqual(track.between(0.5, 2.0).time, (1.0, 2.0))

    def test_compare_figures_distinguish_processed_point_types(self) -> None:
        raw_track = TrajectoryStore(self.parquet_path).load_track(7)
        processed_track = ProcessedTrajectoryStore(self.processed_path).load_track(7)

        time_figure = build_time_series_figure(
            raw_track,
            processed_track=processed_track,
            data_view="compare",
        )
        spatial_figure = build_spatial_figure(
            raw_track,
            processed_track=processed_track,
            data_view="compare",
        )

        time_names = {trace.name for trace in time_figure.data}
        spatial_names = {trace.name for trace in spatial_figure.data}
        self.assertIn("Raw reports", time_names)
        self.assertIn("Observed 1 Hz points", time_names)
        self.assertIn("Interpolated points", time_names)
        self.assertIn("Suspicious transition endpoint", time_names)
        self.assertIn("Standardized trajectory", spatial_names)
        raw_line = next(
            trace for trace in spatial_figure.data if trace.name == "Raw reports"
        )
        processed_line = next(
            trace
            for trace in spatial_figure.data
            if trace.name == "Standardized trajectory"
        )
        processed_points = next(
            trace
            for trace in spatial_figure.data
            if trace.name == "Observed 1 Hz points"
        )
        self.assertEqual(raw_line.mode, "lines+markers")
        self.assertEqual(raw_line.line.color, "#315A9E")
        self.assertGreaterEqual(raw_line.marker.size, 4.5)
        self.assertEqual(processed_line.line.color, "#D97706")
        self.assertEqual(processed_points.marker.color, "#D97706")
        self.assertGreaterEqual(processed_points.marker.size, 5)
        self.assertIn(None, processed_line.x)

    def test_figures_are_created_for_track(self) -> None:
        track = TrajectoryStore(self.parquet_path).load_track(7)

        time_figure = build_time_series_figure(track)
        spatial_figure = build_spatial_figure(track)

        self.assertEqual(len(time_figure.data), 4)
        self.assertEqual(len(spatial_figure.data), 3)
        self.assertIn(None, time_figure.data[0].x)
        self.assertEqual(time_figure.data[0].mode, "lines+markers")
        self.assertEqual(spatial_figure.data[0].mode, "lines+markers")
        self.assertEqual(spatial_figure.data[2].name, "Current position")
        self.assertEqual(len(time_figure.layout.shapes), 0)
        self.assertEqual(
            time_figure.layout.yaxis3.title.text,
            "Geometric altitude (m)",
        )
        self.assertEqual(
            time_figure.layout.yaxis4.title.text,
            "Barometric altitude (m)",
        )
        self.assertEqual(
            time_figure.layout.xaxis4.title.text,
            "Relative time (s)",
        )

    def test_sampling_interval_figure_uses_full_time_differences(self) -> None:
        track = TrajectoryStore(self.parquet_path).load_track(7)

        figure = build_sampling_interval_figure(track)

        self.assertEqual(tuple(figure.data[0].x), (1.0, 2.0, 20.0))
        self.assertEqual(tuple(figure.data[0].y), (1.0, 1.0, 18.0))
        self.assertEqual(figure.layout.xaxis.title.text, "Relative time (s)")
        self.assertEqual(figure.layout.yaxis.title.text, "Sampling interval Δt (s)")

    def test_sampling_distribution_separates_regular_intervals_and_gaps(self) -> None:
        track = TrajectoryStore(self.parquet_path).load_track(7)

        figure = build_sampling_distribution_figure(track)

        self.assertEqual(figure.data[0].type, "bar")
        self.assertEqual(sum(figure.data[0].customdata), 2)
        self.assertEqual(figure.data[1].name, "Long gaps (>2 s)")
        self.assertEqual(sum(figure.data[1].customdata), 1)
        self.assertEqual(tuple(figure.layout.xaxis.range), (0, 2.0))
        self.assertEqual(figure.layout.xaxis.title.text, "Sampling interval Δt (s)")
        self.assertEqual(figure.layout.yaxis.title.text, "Probability density")


if __name__ == "__main__":
    unittest.main()
