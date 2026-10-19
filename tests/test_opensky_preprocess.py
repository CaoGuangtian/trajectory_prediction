from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajectory_prediction.preprocess.config import load_config
from trajectory_prediction.preprocess.deduplicate import DeduplicatedPoint
from trajectory_prediction.preprocess.resample import resample_aircraft
from trajectory_prediction.preprocess.segment import split_runs_on_speed
from trajectory_prediction.preprocess.split import build_aircraft_split


def point(
    timestamp: float,
    lon: float = 8.0,
    lat: float = 50.0,
    geo_altitude: float = 1000.0,
    baro_altitude: float | None = 990.0,
) -> DeduplicatedPoint:
    return DeduplicatedPoint(
        aircraft=1,
        timestamp=timestamp,
        longitude=lon,
        latitude=lat,
        geo_altitude=geo_altitude,
        baro_altitude=baro_altitude,
        num_measurements=4,
        source_record_count=1,
    )


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            PROJECT_ROOT
            / "configs"
            / "preprocessing"
            / "opensky_standardized_1hz_v2.yaml"
        )

    def test_short_gap_is_interpolated(self) -> None:
        runs = resample_aircraft([point(0.1), point(3.1)], 1, 10.0, 2)
        self.assertEqual(len(runs), 1)
        self.assertEqual([item.timestamp for item in runs[0].points], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(
            [item.is_interpolated for item in runs[0].points],
            [False, True, True, False],
        )

    def test_gap_above_interpolation_limit_starts_new_run(self) -> None:
        runs = resample_aircraft([point(0.1), point(4.1)], 1, 10.0, 2)
        self.assertEqual([len(run.points) for run in runs], [1, 1])
        self.assertEqual([run.trajectory_index for run in runs], [1, 1])

    def test_macro_gap_starts_new_trajectory(self) -> None:
        runs = resample_aircraft([point(0.1), point(12.1)], 1, 10.0, 2)
        self.assertEqual([run.trajectory_index for run in runs], [1, 2])

    def test_configured_twenty_second_gap_is_interpolated_without_split(self) -> None:
        runs = resample_aircraft(
            [point(0.1), point(20.1)],
            self.config.sampling_interval_seconds,
            self.config.macro_gap_seconds,
            self.config.max_interpolation_missing_steps,
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0].points), 21)
        self.assertEqual(runs[0].trajectory_index, 1)
        self.assertEqual(sum(item.is_interpolated for item in runs[0].points), 19)

    def test_configured_gap_above_twenty_seconds_starts_new_trajectory(self) -> None:
        runs = resample_aircraft(
            [point(0.1), point(21.1)],
            self.config.sampling_interval_seconds,
            self.config.macro_gap_seconds,
            self.config.max_interpolation_missing_steps,
        )

        self.assertEqual([len(run.points) for run in runs], [1, 1])
        self.assertEqual([run.trajectory_index for run in runs], [1, 2])

    def test_strong_speed_splits_run(self) -> None:
        runs = resample_aircraft([point(0.1), point(1.1, lon=8.02)], 1, 10.0, 2)
        segments, stats = split_runs_on_speed(runs, 1, 400.0, 600.0, 1)
        self.assertEqual(len(segments), 2)
        self.assertEqual(stats.strong_anomaly_breaks, 1)

    def test_segments_shorter_than_two_minutes_are_discarded(self) -> None:
        short_run = resample_aircraft(
            [point(index + 0.1) for index in range(120)],
            1,
            20.0,
            19,
        )
        exact_run = resample_aircraft(
            [point(index + 0.1) for index in range(121)],
            1,
            20.0,
            19,
        )

        short_segments, short_stats = split_runs_on_speed(
            short_run,
            1,
            400.0,
            600.0,
            2,
            120.0,
        )
        exact_segments, exact_stats = split_runs_on_speed(
            exact_run,
            1,
            400.0,
            600.0,
            2,
            120.0,
        )

        self.assertEqual(short_segments, [])
        self.assertEqual(short_stats.short_segments_discarded, 1)
        self.assertEqual(len(exact_segments), 1)
        self.assertEqual(exact_stats.short_segments_discarded, 0)

    def test_aircraft_split_is_stable_and_disjoint(self) -> None:
        first = build_aircraft_split(list(range(1, 101)), self.config)
        second = build_aircraft_split(list(reversed(range(1, 101))), self.config)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(1, 101)))
        self.assertEqual(list(first.values()).count("train"), 70)
        self.assertEqual(list(first.values()).count("validation"), 15)
        self.assertEqual(list(first.values()).count("test"), 15)

    def test_observed_altitudes_are_aggregated_but_not_motion_smoothed(self) -> None:
        runs = resample_aircraft(
            [
                point(0.1, geo_altitude=1000.0, baro_altitude=990.0),
                point(0.8, geo_altitude=1008.0, baro_altitude=998.0),
                point(1.1, geo_altitude=900.0, baro_altitude=890.0),
            ],
            1,
            10.0,
            2,
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual(
            [item.geo_altitude for item in runs[0].points],
            [1004.0, 900.0],
        )
        self.assertEqual(
            [item.baro_altitude for item in runs[0].points],
            [994.0, 890.0],
        )
        self.assertEqual(
            [item.is_interpolated for item in runs[0].points],
            [False, False],
        )


if __name__ == "__main__":
    unittest.main()
