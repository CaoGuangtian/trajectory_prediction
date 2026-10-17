#!/usr/bin/env python3
"""Prepare a compact two-hour realtime replay CSV from OpenSky LocaRDS data.

The input file is scanned twice and never loaded as a whole.  The script keeps
only trusted, continuous aircraft segments, resamples them to 1 Hz, applies
lightweight smoothing, translates the scene to China, and schedules a synthetic
two-hour replay containing 10-20 concurrently active aircraft.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_INPUT_DIR = Path("data/raw/openskyDataset/subset_1")
DEFAULT_OUTPUT_DIR = Path("data/processed/civil_data_for_realtime_web_show")
MAIN_COLUMNS = (
    "id",
    "timeAtServer",
    "aircraft",
    "latitude",
    "longitude",
    "baroAltitude",
    "geoAltitude",
    "numMeasurements",
    "measurements",
)
OUTPUT_COLUMNS = ("P1", "GP", "JD", "WD", "H", "Time")
TIME_FORMAT = "%Y%m%d %H:%M:%S.%f"
EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEGREE_LAT = 111_320.0
SG11_POLY2_WEIGHTS = (-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36)
SG11_POLY2_DIVISOR = 429.0


@dataclass(slots=True)
class RawPoint:
    timestamp: float
    lon: float
    lat: float
    alt: float
    measurements: int


@dataclass(slots=True)
class SegmentSummary:
    aircraft: int
    segment_index: int
    start: float
    end: float
    point_count: int
    duplicate_count: int
    mean_measurements: float
    max_gap_seconds: float
    total_distance_m: float
    mean_speed_mps: float
    excessive_speed_fraction: float
    max_altitude_step_m: float
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    score: float = 0.0
    eligible: bool = False
    rejection_reason: str = ""

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SegmentBuilder:
    aircraft: int
    segment_index: int
    start: float
    end: float
    previous_time: float
    previous_lon: float
    previous_lat: float
    previous_alt: float
    point_count: int = 1
    duplicate_count: int = 0
    measurements_sum: int = 0
    max_gap_seconds: float = 0.0
    total_distance_m: float = 0.0
    speed_sample_count: int = 0
    excessive_speed_count: int = 0
    max_altitude_step_m: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0
    min_lat: float = 0.0
    max_lat: float = 0.0

    @classmethod
    def from_point(
        cls, aircraft: int, segment_index: int, point: RawPoint
    ) -> "SegmentBuilder":
        return cls(
            aircraft=aircraft,
            segment_index=segment_index,
            start=point.timestamp,
            end=point.timestamp,
            previous_time=point.timestamp,
            previous_lon=point.lon,
            previous_lat=point.lat,
            previous_alt=point.alt,
            measurements_sum=point.measurements,
            min_lon=point.lon,
            max_lon=point.lon,
            min_lat=point.lat,
            max_lat=point.lat,
        )

    def add(self, point: RawPoint, excessive_speed_mps: float) -> None:
        dt = point.timestamp - self.previous_time
        if dt == 0:
            self.duplicate_count += 1
            return
        if dt < 0:
            raise ValueError(
                f"aircraft {self.aircraft} time moved backwards: "
                f"{self.previous_time} -> {point.timestamp}"
            )

        distance = haversine_m(
            self.previous_lon,
            self.previous_lat,
            point.lon,
            point.lat,
        )
        speed = distance / dt
        self.total_distance_m += distance
        self.speed_sample_count += 1
        if speed > excessive_speed_mps:
            self.excessive_speed_count += 1
        self.max_altitude_step_m = max(
            self.max_altitude_step_m,
            abs(point.alt - self.previous_alt),
        )
        self.max_gap_seconds = max(self.max_gap_seconds, dt)
        self.end = point.timestamp
        self.previous_time = point.timestamp
        self.previous_lon = point.lon
        self.previous_lat = point.lat
        self.previous_alt = point.alt
        self.point_count += 1
        self.measurements_sum += point.measurements
        self.min_lon = min(self.min_lon, point.lon)
        self.max_lon = max(self.max_lon, point.lon)
        self.min_lat = min(self.min_lat, point.lat)
        self.max_lat = max(self.max_lat, point.lat)

    def finish(self) -> SegmentSummary:
        duration = max(0.0, self.end - self.start)
        return SegmentSummary(
            aircraft=self.aircraft,
            segment_index=self.segment_index,
            start=self.start,
            end=self.end,
            point_count=self.point_count,
            duplicate_count=self.duplicate_count,
            mean_measurements=self.measurements_sum / max(self.point_count, 1),
            max_gap_seconds=self.max_gap_seconds,
            total_distance_m=self.total_distance_m,
            mean_speed_mps=self.total_distance_m / duration if duration > 0 else 0.0,
            excessive_speed_fraction=(
                self.excessive_speed_count / max(self.speed_sample_count, 1)
            ),
            max_altitude_step_m=self.max_altitude_step_m,
            min_lon=self.min_lon,
            max_lon=self.max_lon,
            min_lat=self.min_lat,
            max_lat=self.max_lat,
        )


@dataclass(slots=True)
class ReplayPoint:
    offset_seconds: int
    lon: float
    lat: float
    alt: float


@dataclass(slots=True)
class Candidate:
    summary: SegmentSummary
    points: list[ReplayPoint]
    coverage_ratio: float
    processed_mean_speed_mps: float
    processed_p99_speed_mps: float
    max_smoothing_correction_m: float
    replay_altitude_flip_rate_per_min: float
    centroid_lon: float
    centroid_lat: float
    final_score: float

    @property
    def duration_seconds(self) -> int:
        return max(0, len(self.points) - 1)


@dataclass(slots=True)
class ScheduledFlight:
    candidate: Candidate
    start_offset: int
    p1: str

    @property
    def end_offset_exclusive(self) -> int:
        return self.start_offset + len(self.candidate.points)


@dataclass(slots=True)
class RuntimeStats:
    scanned_rows_first_pass: int = 0
    scanned_rows_second_pass: int = 0
    malformed_rows_first_pass: int = 0
    malformed_rows_second_pass: int = 0
    trusted_aircraft_count: int = 0
    segment_count: int = 0
    eligible_summary_count: int = 0
    extracted_candidate_count: int = 0
    processed_candidate_count: int = 0
    clustered_candidate_count: int = 0
    scheduled_flight_count: int = 0
    output_row_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact two-hour replay CSV from OpenSky LocaRDS subset_1."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--replay-start",
        default="20250510 08:00:00.000000",
        help=(
            "Synthetic replay start in "
            f"{TIME_FORMAT.replace('%', '%%')!r} format."
        ),
    )
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--min-aircraft", type=int, default=10)
    parser.add_argument("--max-aircraft", type=int, default=20)
    parser.add_argument(
        "--concurrency-step-seconds",
        type=int,
        default=60,
        help="How often to draw a new random concurrency target.",
    )
    parser.add_argument("--seed", type=int, default=20250516)
    parser.add_argument("--gap-seconds", type=float, default=10.0)
    parser.add_argument("--interpolation-gap-seconds", type=int, default=5)
    parser.add_argument("--min-duration-minutes", type=float, default=30.0)
    parser.add_argument("--min-points", type=int, default=900)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.90)
    parser.add_argument("--min-mean-measurements", type=float, default=4.0)
    parser.add_argument("--min-mean-speed-mps", type=float, default=70.0)
    parser.add_argument("--max-mean-speed-mps", type=float, default=350.0)
    parser.add_argument("--max-p99-speed-mps", type=float, default=450.0)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--cluster-radius-km", type=float, default=700.0)
    parser.add_argument("--target-center-lon", type=float, default=112.0)
    parser.add_argument("--target-center-lat", type=float, default=32.0)
    parser.add_argument(
        "--target-bounds",
        default="100,20,124,44",
        help="Allowed lon_min,lat_min,lon_max,lat_max after relocation.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Output CSV name. Default is the replay date, for example 20250510.csv.",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable isolated-point repair and Savitzky-Golay smoothing.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[datetime, int, tuple[float, ...]]:
    try:
        replay_start = datetime.strptime(args.replay_start, TIME_FORMAT)
    except ValueError as exc:
        raise SystemExit(
            f"--replay-start must match {TIME_FORMAT!r}: {args.replay_start!r}"
        ) from exc

    duration_seconds = int(round(args.duration_hours * 3600))
    if duration_seconds < 1:
        raise SystemExit("--duration-hours must produce at least one second.")
    if replay_start.date() != (replay_start + timedelta(seconds=duration_seconds - 1)).date():
        raise SystemExit("Replay must not cross midnight because the target protocol carries HHMMSS only.")
    if not 1 <= args.min_aircraft <= args.max_aircraft:
        raise SystemExit("Require 1 <= --min-aircraft <= --max-aircraft.")
    if args.max_aircraft > 100:
        raise SystemExit("--max-aircraft above 100 is not intended for this demo generator.")
    if args.concurrency_step_seconds < 1:
        raise SystemExit("--concurrency-step-seconds must be positive.")
    if args.gap_seconds <= 0 or args.interpolation_gap_seconds < 0:
        raise SystemExit("Gap thresholds must be non-negative and --gap-seconds positive.")
    if args.min_duration_minutes <= 0 or args.min_points < 2:
        raise SystemExit("Duration and point thresholds must be positive.")
    if not 0 < args.min_coverage_ratio <= 1:
        raise SystemExit("--min-coverage-ratio must be in (0, 1].")
    if args.candidate_limit < args.max_aircraft:
        raise SystemExit("--candidate-limit must be at least --max-aircraft.")
    if args.cluster_radius_km <= 0:
        raise SystemExit("--cluster-radius-km must be positive.")

    try:
        bounds = tuple(float(part.strip()) for part in args.target_bounds.split(","))
    except ValueError as exc:
        raise SystemExit("--target-bounds must contain four comma-separated numbers.") from exc
    if len(bounds) != 4:
        raise SystemExit("--target-bounds must be lon_min,lat_min,lon_max,lat_max.")
    lon_min, lat_min, lon_max, lat_max = bounds
    if not (100 <= lon_min < lon_max < 180 and 0 < lat_min < lat_max < 90):
        raise SystemExit(
            "Target bounds must be positive, encoder-compatible China-region coordinates."
        )
    if not (lon_min < args.target_center_lon < lon_max):
        raise SystemExit("--target-center-lon must be inside --target-bounds.")
    if not (lat_min < args.target_center_lat < lat_max):
        raise SystemExit("--target-center-lat must be inside --target-bounds.")
    return replay_start, duration_seconds, bounds


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, (lon1, lat1, lon2, lat2)
    )
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_trusted_aircraft(path: Path) -> set[int]:
    trusted: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["aircraft", "trusted"]:
            raise ValueError(f"Unexpected aircraft metadata header: {reader.fieldnames}")
        for row in reader:
            if row["trusted"].strip().upper() == "TRUE":
                trusted.add(int(row["aircraft"]))
    return trusted


def iter_main_rows(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield the first eight CSV fields without parsing the huge measurements field."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if tuple(header) != MAIN_COLUMNS:
            raise ValueError(f"Unexpected main CSV header: {header}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split(",", 8)
            if len(fields) != 9:
                yield line_number, []
                continue
            yield line_number, fields


def parse_point(fields: list[str]) -> tuple[int, RawPoint] | None:
    aircraft = int(fields[2])
    timestamp = float(fields[1])
    lat = float(fields[3])
    lon = float(fields[4])
    alt = float(fields[6])
    measurements = int(fields[7])
    if not all(math.isfinite(value) for value in (timestamp, lat, lon, alt)):
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return aircraft, RawPoint(timestamp, lon, lat, alt, measurements)


def scan_segment_summaries(
    main_path: Path,
    trusted: set[int],
    args: argparse.Namespace,
    stats: RuntimeStats,
) -> list[SegmentSummary]:
    builders: dict[int, SegmentBuilder] = {}
    next_segment_index: dict[int, int] = {}
    summaries: list[SegmentSummary] = []

    for _, fields in iter_main_rows(main_path):
        stats.scanned_rows_first_pass += 1
        if not fields:
            stats.malformed_rows_first_pass += 1
            continue
        try:
            aircraft = int(fields[2])
            if aircraft not in trusted:
                continue
            parsed = parse_point(fields)
            if parsed is None:
                continue
            _, point = parsed
        except (TypeError, ValueError):
            stats.malformed_rows_first_pass += 1
            continue

        builder = builders.get(aircraft)
        if builder is None:
            segment_index = next_segment_index.get(aircraft, 1)
            builders[aircraft] = SegmentBuilder.from_point(
                aircraft, segment_index, point
            )
            next_segment_index[aircraft] = segment_index + 1
            continue

        if point.timestamp - builder.previous_time > args.gap_seconds:
            summaries.append(builder.finish())
            segment_index = next_segment_index[aircraft]
            builders[aircraft] = SegmentBuilder.from_point(
                aircraft, segment_index, point
            )
            next_segment_index[aircraft] = segment_index + 1
            continue

        try:
            builder.add(point, args.max_p99_speed_mps)
        except ValueError:
            stats.malformed_rows_first_pass += 1

    summaries.extend(builder.finish() for builder in builders.values())
    stats.segment_count = len(summaries)
    return summaries


def assess_summary(summary: SegmentSummary, args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if summary.duration_seconds < args.min_duration_minutes * 60:
        reasons.append(f"duration<{args.min_duration_minutes:g}min")
    if summary.point_count < args.min_points:
        reasons.append(f"points<{args.min_points}")
    if summary.mean_measurements < args.min_mean_measurements:
        reasons.append(f"mean_measurements<{args.min_mean_measurements:g}")
    if summary.mean_speed_mps < args.min_mean_speed_mps:
        reasons.append(f"mean_speed<{args.min_mean_speed_mps:g}mps")
    if summary.mean_speed_mps > args.max_mean_speed_mps:
        reasons.append(f"mean_speed>{args.max_mean_speed_mps:g}mps")
    if summary.excessive_speed_fraction > 0.02:
        reasons.append("excessive_speed_fraction>0.02")

    summary.score = (
        min(summary.duration_seconds / 60.0, 60.0) * 1.8
        + min(summary.mean_measurements, 15.0) * 4.0
        + min(summary.total_distance_m / 1000.0, 800.0) * 0.05
        - summary.excessive_speed_fraction * 1000.0
        - max(0.0, summary.max_altitude_step_m - 300.0) * 0.03
    )
    summary.eligible = not reasons
    summary.rejection_reason = ";".join(reasons)


def choose_best_summaries(
    summaries: list[SegmentSummary], args: argparse.Namespace
) -> list[SegmentSummary]:
    for summary in summaries:
        assess_summary(summary, args)

    best_by_aircraft: dict[int, SegmentSummary] = {}
    for summary in summaries:
        if not summary.eligible:
            continue
        current = best_by_aircraft.get(summary.aircraft)
        if current is None or (summary.score, summary.duration_seconds) > (
            current.score,
            current.duration_seconds,
        ):
            best_by_aircraft[summary.aircraft] = summary

    return sorted(
        best_by_aircraft.values(),
        key=lambda item: (-item.score, -item.duration_seconds, item.aircraft),
    )[: args.candidate_limit]


def extract_candidate_points(
    main_path: Path,
    selected: list[SegmentSummary],
    stats: RuntimeStats,
) -> dict[int, list[RawPoint]]:
    selected_by_aircraft = {summary.aircraft: summary for summary in selected}
    points_by_aircraft: dict[int, list[RawPoint]] = {
        summary.aircraft: [] for summary in selected
    }

    for _, fields in iter_main_rows(main_path):
        stats.scanned_rows_second_pass += 1
        if not fields:
            stats.malformed_rows_second_pass += 1
            continue
        try:
            aircraft = int(fields[2])
            summary = selected_by_aircraft.get(aircraft)
            if summary is None:
                continue
            timestamp = float(fields[1])
            if timestamp < summary.start or timestamp > summary.end:
                continue
            parsed = parse_point(fields)
            if parsed is None:
                continue
            points_by_aircraft[aircraft].append(parsed[1])
        except (TypeError, ValueError):
            stats.malformed_rows_second_pass += 1

    return points_by_aircraft


def median_point_for_second(second: int, points: list[RawPoint]) -> ReplayPoint:
    return ReplayPoint(
        offset_seconds=second,
        lon=statistics.median(point.lon for point in points),
        lat=statistics.median(point.lat for point in points),
        alt=statistics.median(point.alt for point in points),
    )


def resample_longest_run(
    raw_points: list[RawPoint], interpolation_gap_seconds: int
) -> tuple[list[ReplayPoint], float]:
    bins: dict[int, list[RawPoint]] = {}
    for point in raw_points:
        bins.setdefault(math.floor(point.timestamp), []).append(point)
    if len(bins) < 2:
        return [], 0.0

    known = {
        second: median_point_for_second(second, points)
        for second, points in bins.items()
    }
    first_second = min(known)
    last_second = max(known)
    expected_count = last_second - first_second + 1
    filled: dict[int, ReplayPoint] = dict(known)
    known_seconds = sorted(known)

    for left_second, right_second in zip(known_seconds, known_seconds[1:]):
        missing_count = right_second - left_second - 1
        if missing_count <= 0 or missing_count > interpolation_gap_seconds:
            continue
        left = known[left_second]
        right = known[right_second]
        span = right_second - left_second
        for second in range(left_second + 1, right_second):
            ratio = (second - left_second) / span
            filled[second] = ReplayPoint(
                offset_seconds=second,
                lon=left.lon + (right.lon - left.lon) * ratio,
                lat=left.lat + (right.lat - left.lat) * ratio,
                alt=left.alt + (right.alt - left.alt) * ratio,
            )

    runs: list[list[ReplayPoint]] = []
    current: list[ReplayPoint] = []
    previous_second: int | None = None
    for second in sorted(filled):
        if previous_second is not None and second != previous_second + 1:
            if current:
                runs.append(current)
            current = []
        current.append(filled[second])
        previous_second = second
    if current:
        runs.append(current)
    if not runs:
        return [], 0.0

    longest = max(runs, key=len)
    run_start = longest[0].offset_seconds
    normalized = [
        ReplayPoint(
            offset_seconds=point.offset_seconds - run_start,
            lon=point.lon,
            lat=point.lat,
            alt=point.alt,
        )
        for point in longest
    ]
    coverage_ratio = len(known) / max(expected_count, 1)
    return normalized, coverage_ratio


def lonlat_to_local_xy(
    lon: float, lat: float, origin_lon: float, origin_lat: float
) -> tuple[float, float]:
    cosine = max(0.1, math.cos(math.radians(origin_lat)))
    return (
        (lon - origin_lon) * METERS_PER_DEGREE_LAT * cosine,
        (lat - origin_lat) * METERS_PER_DEGREE_LAT,
    )


def local_xy_to_lonlat(
    x: float, y: float, origin_lon: float, origin_lat: float
) -> tuple[float, float]:
    cosine = max(0.1, math.cos(math.radians(origin_lat)))
    return (
        origin_lon + x / (METERS_PER_DEGREE_LAT * cosine),
        origin_lat + y / METERS_PER_DEGREE_LAT,
    )


def repair_isolated_points(
    xs: list[float], ys: list[float], alts: list[float]
) -> tuple[list[float], list[float], list[float]]:
    clean_x = list(xs)
    clean_y = list(ys)
    clean_alt = list(alts)
    for _ in range(2):
        next_x = list(clean_x)
        next_y = list(clean_y)
        next_alt = list(clean_alt)
        for index in range(1, len(clean_x) - 1):
            expected_x = (clean_x[index - 1] + clean_x[index + 1]) / 2
            expected_y = (clean_y[index - 1] + clean_y[index + 1]) / 2
            residual = math.hypot(
                clean_x[index] - expected_x,
                clean_y[index] - expected_y,
            )
            if residual > 1_500.0:
                next_x[index] = expected_x
                next_y[index] = expected_y
            expected_alt = (clean_alt[index - 1] + clean_alt[index + 1]) / 2
            if abs(clean_alt[index] - expected_alt) > 400.0:
                next_alt[index] = expected_alt
        clean_x, clean_y, clean_alt = next_x, next_y, next_alt
    return clean_x, clean_y, clean_alt


def savgol_11_poly2(values: list[float]) -> list[float]:
    if len(values) < len(SG11_POLY2_WEIGHTS):
        return list(values)
    result = list(values)
    radius = len(SG11_POLY2_WEIGHTS) // 2
    for center in range(radius, len(values) - radius):
        window = values[center - radius : center + radius + 1]
        result[center] = sum(
            weight * value for weight, value in zip(SG11_POLY2_WEIGHTS, window)
        ) / SG11_POLY2_DIVISOR
    return result


def smooth_points(
    points: list[ReplayPoint], enabled: bool
) -> tuple[list[ReplayPoint], float]:
    if not points:
        return [], 0.0
    origin_lon = points[0].lon
    origin_lat = points[0].lat
    xy = [
        lonlat_to_local_xy(point.lon, point.lat, origin_lon, origin_lat)
        for point in points
    ]
    original_x = [value[0] for value in xy]
    original_y = [value[1] for value in xy]
    original_alt = [point.alt for point in points]
    if enabled:
        clean_x, clean_y, clean_alt = repair_isolated_points(
            original_x, original_y, original_alt
        )
        smooth_x = savgol_11_poly2(clean_x)
        smooth_y = savgol_11_poly2(clean_y)
        smooth_alt = savgol_11_poly2(clean_alt)
    else:
        smooth_x, smooth_y, smooth_alt = original_x, original_y, original_alt

    result: list[ReplayPoint] = []
    max_correction = 0.0
    for source, x, y, alt, raw_x, raw_y in zip(
        points, smooth_x, smooth_y, smooth_alt, original_x, original_y
    ):
        lon, lat = local_xy_to_lonlat(x, y, origin_lon, origin_lat)
        result.append(ReplayPoint(source.offset_seconds, lon, lat, max(1.0, alt)))
        max_correction = max(max_correction, math.hypot(x - raw_x, y - raw_y))
    return result, max_correction


def replay_altitude_flip_rate(points: list[ReplayPoint]) -> float:
    visible = [math.floor(point.alt) // 100 * 100 for point in points]
    previous_sign = 0
    flips = 0
    for left, right in zip(visible, visible[1:]):
        delta = right - left
        if abs(delta) < 100:
            continue
        sign = 1 if delta > 0 else -1
        if previous_sign and sign != previous_sign:
            flips += 1
        previous_sign = sign
    minutes = max((len(points) - 1) / 60.0, 1 / 60.0)
    return flips / minutes


def build_candidate(
    summary: SegmentSummary,
    raw_points: list[RawPoint],
    args: argparse.Namespace,
) -> tuple[Candidate | None, str]:
    points, coverage_ratio = resample_longest_run(
        raw_points, args.interpolation_gap_seconds
    )
    minimum_seconds = int(round(args.min_duration_minutes * 60))
    if len(points) < minimum_seconds + 1:
        return None, "resampled_duration_too_short"
    if coverage_ratio < args.min_coverage_ratio:
        return None, f"coverage<{args.min_coverage_ratio:g}"

    points, max_correction = smooth_points(points, enabled=not args.no_smoothing)
    speeds = [
        haversine_m(left.lon, left.lat, right.lon, right.lat)
        for left, right in zip(points, points[1:])
    ]
    mean_speed = statistics.fmean(speeds) if speeds else 0.0
    p99_speed = percentile(speeds, 0.99)
    if mean_speed < args.min_mean_speed_mps:
        return None, f"processed_mean_speed<{args.min_mean_speed_mps:g}mps"
    if mean_speed > args.max_mean_speed_mps:
        return None, f"processed_mean_speed>{args.max_mean_speed_mps:g}mps"
    if p99_speed > args.max_p99_speed_mps:
        return None, f"processed_p99_speed>{args.max_p99_speed_mps:g}mps"

    altitude_flip_rate = replay_altitude_flip_rate(points)
    centroid_lon = statistics.fmean(point.lon for point in points)
    centroid_lat = statistics.fmean(point.lat for point in points)
    final_score = (
        summary.score
        + min((len(points) - 1) / 60.0, 60.0)
        + coverage_ratio * 30.0
        - p99_speed / 100.0
        - altitude_flip_rate * 8.0
        - max_correction / 1000.0
    )
    return (
        Candidate(
            summary=summary,
            points=points,
            coverage_ratio=coverage_ratio,
            processed_mean_speed_mps=mean_speed,
            processed_p99_speed_mps=p99_speed,
            max_smoothing_correction_m=max_correction,
            replay_altitude_flip_rate_per_min=altitude_flip_rate,
            centroid_lon=centroid_lon,
            centroid_lat=centroid_lat,
            final_score=final_score,
        ),
        "",
    )


def choose_compact_cluster(candidates: list[Candidate], radius_km: float) -> list[Candidate]:
    if not candidates:
        return []
    radius_m = radius_km * 1000.0
    best: list[Candidate] = []
    best_key = (-1, -1.0)
    for center in candidates:
        members = [
            candidate
            for candidate in candidates
            if haversine_m(
                center.centroid_lon,
                center.centroid_lat,
                candidate.centroid_lon,
                candidate.centroid_lat,
            )
            <= radius_m
        ]
        key = (
            sum(candidate.duration_seconds for candidate in members),
            sum(candidate.final_score for candidate in members),
        )
        if key > best_key:
            best_key = key
            best = members
    return sorted(best, key=lambda item: (-item.final_score, item.summary.aircraft))


def generate_random_p1(rng: random.Random, used: set[str]) -> str:
    """Return a reproducible, globally unique random 4-6 digit replay ID."""
    for _ in range(10_000):
        digits = rng.choice((4, 5, 6))
        value = str(rng.randint(10 ** (digits - 1), 10**digits - 1))
        if value not in used:
            used.add(value)
            return value
    raise RuntimeError("Unable to generate a unique random P1.")


def schedule_candidates(
    candidates: list[Candidate],
    duration_seconds: int,
    args: argparse.Namespace,
) -> tuple[list[ScheduledFlight], list[int], list[int]]:
    rng = random.Random(args.seed)
    randomized = [(candidate, rng.random()) for candidate in candidates]
    randomized.sort(key=lambda item: (-item[0].final_score, item[1]))
    pool = [item[0] for item in randomized]
    active: list[ScheduledFlight] = []
    scheduled: list[ScheduledFlight] = []
    used_p1: set[str] = set()
    concurrency: list[int] = []
    targets: list[int] = []
    desired = rng.randint(args.min_aircraft, args.max_aircraft)

    total_available_seconds = sum(len(candidate.points) for candidate in pool)
    minimum_required_seconds = args.min_aircraft * duration_seconds
    if len(pool) < args.min_aircraft or total_available_seconds < minimum_required_seconds:
        raise RuntimeError(
            "Not enough high-quality candidate duration for the requested replay: "
            f"candidates={len(pool)}, available_aircraft_seconds={total_available_seconds}, "
            f"minimum_required={minimum_required_seconds}."
        )

    for second in range(duration_seconds):
        active = [flight for flight in active if flight.end_offset_exclusive > second]
        if second and second % args.concurrency_step_seconds == 0:
            desired = max(
                args.min_aircraft,
                min(args.max_aircraft, desired + rng.randint(-3, 3)),
            )
        targets.append(desired)

        launch_target = max(args.min_aircraft, desired)
        while len(active) < launch_target:
            if not pool:
                raise RuntimeError(
                    f"Candidate pool exhausted at replay second {second}; "
                    "relax filters or increase --candidate-limit."
                )
            lookahead = min(8, len(pool))
            choice_index = rng.randrange(lookahead)
            candidate = pool.pop(choice_index)
            flight = ScheduledFlight(
                candidate=candidate,
                start_offset=second,
                p1=generate_random_p1(rng, used_p1),
            )
            active.append(flight)
            scheduled.append(flight)

        if not args.min_aircraft <= len(active) <= args.max_aircraft:
            raise RuntimeError(
                f"Concurrency invariant failed at second {second}: {len(active)}"
            )
        concurrency.append(len(active))

    return scheduled, concurrency, targets


def source_scene_center(scheduled: list[ScheduledFlight]) -> tuple[float, float]:
    lons = [
        point.lon
        for flight in scheduled
        for point in flight.candidate.points
    ]
    lats = [
        point.lat
        for flight in scheduled
        for point in flight.candidate.points
    ]
    return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2


def relocate_to_china(
    lon: float,
    lat: float,
    source_center_lon: float,
    source_center_lat: float,
    target_center_lon: float,
    target_center_lat: float,
) -> tuple[float, float]:
    x, y = lonlat_to_local_xy(lon, lat, source_center_lon, source_center_lat)
    return local_xy_to_lonlat(x, y, target_center_lon, target_center_lat)


def decimal_degrees_to_packed_dms(value: float) -> str:
    if value < 0:
        raise ValueError("The replay encoder does not support negative coordinates.")
    # The target protocol keeps only DDD.MMSS/DDD.MMSS with one digit for
    # tens-of-seconds. Round in angular seconds first so 59m59s correctly
    # carries into the next degree instead of producing an invalid .596 value.
    total_seconds = (Decimal(str(value)) * Decimal(3600)).quantize(
        Decimal("1E1"), rounding=ROUND_HALF_UP
    )
    degrees = int(total_seconds // Decimal(3600))
    remaining = total_seconds - Decimal(degrees * 3600)
    minutes = int(remaining // Decimal(60))
    seconds = int(remaining - Decimal(minutes * 60))
    packed = Decimal(degrees) + Decimal(minutes) / 100 + Decimal(seconds) / 10_000
    return format(packed.quantize(Decimal("0.001")), ".3f")


def protocol_quantize_packed(raw: str, *, longitude: bool) -> str:
    value = Decimal(raw).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    fractional = format(value, ".3f").split(".", 1)[1]
    minutes = int(fractional[:2])
    tens_of_seconds = int(fractional[2])
    if minutes >= 60 or tens_of_seconds >= 6:
        raise ValueError(f"Packed coordinate has invalid minutes/seconds: {raw}")
    digits = format(value, ".3f").replace(".", "")
    if longitude:
        if value < 100:
            raise ValueError(f"Longitude must be at least 100 degrees, got {value}.")
        digits = digits[1:]
    expected = 5
    if len(digits) > expected:
        raise ValueError(f"Packed coordinate does not fit {expected} digits: {raw}")
    return digits.zfill(expected)


def validate_output_row(row: dict[str, str]) -> None:
    p1 = row["P1"]
    if not p1.isdigit() or not 4 <= len(p1) <= 6:
        raise ValueError(f"P1 must contain 4-6 digits: {p1!r}")
    if row["GP"] and (not row["GP"].isdigit() or not 3 <= len(row["GP"]) <= 6):
        raise ValueError(f"Invalid GP: {row['GP']!r}")
    protocol_quantize_packed(row["JD"], longitude=True)
    protocol_quantize_packed(row["WD"], longitude=False)
    height = int(Decimal(row["H"]))
    encoded_height = height // 100 if height >= 100 else 900 + height
    if not 0 <= encoded_height <= 999:
        raise ValueError(f"Height cannot be encoded: {row['H']!r}")
    datetime.strptime(row["Time"], TIME_FORMAT)


def build_output_rows(
    scheduled: list[ScheduledFlight],
    replay_start: datetime,
    duration_seconds: int,
    bounds: tuple[float, ...],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], tuple[float, float], tuple[float, float, float, float]]:
    source_center_lon, source_center_lat = source_scene_center(scheduled)
    lon_min, lat_min, lon_max, lat_max = bounds
    rows_with_keys: list[tuple[int, str, dict[str, str]]] = []
    output_lons: list[float] = []
    output_lats: list[float] = []

    for flight in scheduled:
        for point in flight.candidate.points:
            replay_second = flight.start_offset + point.offset_seconds
            if replay_second >= duration_seconds:
                break
            lon, lat = relocate_to_china(
                point.lon,
                point.lat,
                source_center_lon,
                source_center_lat,
                args.target_center_lon,
                args.target_center_lat,
            )
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                raise RuntimeError(
                    "Relocated point falls outside --target-bounds: "
                    f"aircraft={flight.candidate.summary.aircraft}, lon={lon:.6f}, "
                    f"lat={lat:.6f}. Reduce --cluster-radius-km or adjust the target area."
                )
            timestamp = replay_start + timedelta(seconds=replay_second)
            row = {
                "P1": flight.p1,
                "GP": "",
                "JD": decimal_degrees_to_packed_dms(lon),
                "WD": decimal_degrees_to_packed_dms(lat),
                "H": str(max(1, int(round(point.alt)))),
                "Time": timestamp.strftime(TIME_FORMAT),
            }
            validate_output_row(row)
            rows_with_keys.append((replay_second, flight.p1, row))
            output_lons.append(lon)
            output_lats.append(lat)

    rows_with_keys.sort(key=lambda item: (item[0], item[1]))
    rows = [item[2] for item in rows_with_keys]
    output_bounds = (
        min(output_lons),
        min(output_lats),
        max(output_lons),
        max(output_lats),
    )
    return rows, (source_center_lon, source_center_lat), output_bounds


def write_replay_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_report(
    path: Path,
    summaries: list[SegmentSummary],
    processing_rejections: dict[int, str],
    clustered_aircraft: set[int],
    scheduled_aircraft: set[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "aircraft",
        "segment_index",
        "start",
        "end",
        "duration_seconds",
        "point_count",
        "duplicate_count",
        "mean_measurements",
        "max_gap_seconds",
        "total_distance_m",
        "mean_speed_mps",
        "excessive_speed_fraction",
        "max_altitude_step_m",
        "score",
        "summary_eligible",
        "rejection_reason",
        "processing_rejection",
        "in_compact_cluster",
        "scheduled",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: (item.aircraft, item.start)):
            writer.writerow(
                {
                    "aircraft": summary.aircraft,
                    "segment_index": summary.segment_index,
                    "start": f"{summary.start:.6f}",
                    "end": f"{summary.end:.6f}",
                    "duration_seconds": f"{summary.duration_seconds:.3f}",
                    "point_count": summary.point_count,
                    "duplicate_count": summary.duplicate_count,
                    "mean_measurements": f"{summary.mean_measurements:.3f}",
                    "max_gap_seconds": f"{summary.max_gap_seconds:.3f}",
                    "total_distance_m": f"{summary.total_distance_m:.3f}",
                    "mean_speed_mps": f"{summary.mean_speed_mps:.3f}",
                    "excessive_speed_fraction": f"{summary.excessive_speed_fraction:.6f}",
                    "max_altitude_step_m": f"{summary.max_altitude_step_m:.3f}",
                    "score": f"{summary.score:.3f}",
                    "summary_eligible": str(summary.eligible).upper(),
                    "rejection_reason": summary.rejection_reason,
                    "processing_rejection": processing_rejections.get(
                        summary.aircraft, ""
                    ),
                    "in_compact_cluster": str(
                        summary.aircraft in clustered_aircraft
                    ).upper(),
                    "scheduled": str(summary.aircraft in scheduled_aircraft).upper(),
                }
            )


def write_schedule_report(path: Path, scheduled: list[ScheduledFlight], replay_start: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "P1",
        "source_aircraft",
        "source_segment_index",
        "source_start",
        "source_end",
        "replay_start",
        "replay_end",
        "duration_seconds",
        "point_count",
        "coverage_ratio",
        "mean_speed_mps",
        "p99_speed_mps",
        "quality_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for flight in sorted(scheduled, key=lambda item: (item.start_offset, item.p1)):
            candidate = flight.candidate
            writer.writerow(
                {
                    "P1": flight.p1,
                    "source_aircraft": candidate.summary.aircraft,
                    "source_segment_index": candidate.summary.segment_index,
                    "source_start": f"{candidate.summary.start:.6f}",
                    "source_end": f"{candidate.summary.end:.6f}",
                    "replay_start": (replay_start + timedelta(seconds=flight.start_offset)).strftime(
                        TIME_FORMAT
                    ),
                    "replay_end": (
                        replay_start
                        + timedelta(seconds=flight.end_offset_exclusive - 1)
                    ).strftime(TIME_FORMAT),
                    "duration_seconds": candidate.duration_seconds,
                    "point_count": len(candidate.points),
                    "coverage_ratio": f"{candidate.coverage_ratio:.6f}",
                    "mean_speed_mps": f"{candidate.processed_mean_speed_mps:.3f}",
                    "p99_speed_mps": f"{candidate.processed_p99_speed_mps:.3f}",
                    "quality_score": f"{candidate.final_score:.3f}",
                }
            )


def write_concurrency_report(
    path: Path,
    replay_start: datetime,
    concurrency: list[int],
    targets: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Time", "active_aircraft_count", "random_target"),
        )
        writer.writeheader()
        for second, (count, target) in enumerate(zip(concurrency, targets)):
            writer.writerow(
                {
                    "Time": (replay_start + timedelta(seconds=second)).strftime(
                        TIME_FORMAT
                    ),
                    "active_aircraft_count": count,
                    "random_target": target,
                }
            )


def main() -> None:
    args = parse_args()
    replay_start, duration_seconds, bounds = validate_args(args)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    main_path = input_dir / "set_1.csv"
    aircraft_path = input_dir / "set_1_aircraft.csv"
    for path in (main_path, aircraft_path):
        if not path.is_file():
            raise SystemExit(f"Required input file does not exist: {path}")

    stats = RuntimeStats()
    trusted = read_trusted_aircraft(aircraft_path)
    stats.trusted_aircraft_count = len(trusted)
    print(f"trusted aircraft: {len(trusted)}")

    print("pass 1/2: scanning segment quality...")
    summaries = scan_segment_summaries(main_path, trusted, args, stats)
    selected_summaries = choose_best_summaries(summaries, args)
    stats.eligible_summary_count = sum(summary.eligible for summary in summaries)
    stats.extracted_candidate_count = len(selected_summaries)
    if len(selected_summaries) < args.min_aircraft:
        raise SystemExit(
            "Too few eligible aircraft after the first pass: "
            f"{len(selected_summaries)}. Relax the selection thresholds."
        )

    print(f"pass 2/2: extracting {len(selected_summaries)} candidate trajectories...")
    raw_points = extract_candidate_points(main_path, selected_summaries, stats)
    candidates: list[Candidate] = []
    processing_rejections: dict[int, str] = {}
    for summary in selected_summaries:
        candidate, reason = build_candidate(summary, raw_points[summary.aircraft], args)
        if candidate is None:
            processing_rejections[summary.aircraft] = reason
        else:
            candidates.append(candidate)
    stats.processed_candidate_count = len(candidates)

    clustered = choose_compact_cluster(candidates, args.cluster_radius_km)
    stats.clustered_candidate_count = len(clustered)
    if len(clustered) < args.min_aircraft:
        raise SystemExit(
            "Too few candidates remain in one compact geographic cluster: "
            f"{len(clustered)}. Increase --cluster-radius-km or relax filters."
        )

    scheduled, concurrency, targets = schedule_candidates(
        clustered, duration_seconds, args
    )
    stats.scheduled_flight_count = len(scheduled)
    rows, source_center, output_bounds = build_output_rows(
        scheduled,
        replay_start,
        duration_seconds,
        bounds,
        args,
    )
    stats.output_row_count = len(rows)

    output_name = args.output_name.strip() or f"{replay_start:%Y%m%d}.csv"
    if Path(output_name).name != output_name or not output_name.lower().endswith(".csv"):
        raise SystemExit("--output-name must be a plain .csv file name.")
    output_path = output_dir / output_name
    report_dir = output_dir / "report"
    write_replay_csv(output_path, rows)

    clustered_aircraft = {candidate.summary.aircraft for candidate in clustered}
    scheduled_aircraft = {
        flight.candidate.summary.aircraft for flight in scheduled
    }
    write_selection_report(
        report_dir / "selection_report.csv",
        summaries,
        processing_rejections,
        clustered_aircraft,
        scheduled_aircraft,
    )
    write_schedule_report(report_dir / "schedule_report.csv", scheduled, replay_start)
    write_concurrency_report(
        report_dir / "concurrency_report.csv",
        replay_start,
        concurrency,
        targets,
    )
    summary_payload = {
        "input_dir": str(input_dir),
        "output_csv": str(output_path),
        "replay_start": replay_start.strftime(TIME_FORMAT),
        "duration_seconds": duration_seconds,
        "random_seed": args.seed,
        "p1_policy": "unique random 4-6 digit number",
        "source_scene_center": {
            "longitude": source_center[0],
            "latitude": source_center[1],
        },
        "target_scene_center": {
            "longitude": args.target_center_lon,
            "latitude": args.target_center_lat,
        },
        "output_bounds": {
            "longitude_min": output_bounds[0],
            "latitude_min": output_bounds[1],
            "longitude_max": output_bounds[2],
            "latitude_max": output_bounds[3],
        },
        "concurrency": {
            "minimum": min(concurrency),
            "maximum": max(concurrency),
            "mean": statistics.fmean(concurrency),
        },
        "stats": asdict(stats),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote replay CSV: {output_path}")
    print(f"rows: {len(rows)}")
    print(
        "concurrency: "
        f"min={min(concurrency)}, max={max(concurrency)}, "
        f"mean={statistics.fmean(concurrency):.2f}"
    )
    print(f"random P1 count: {len({flight.p1 for flight in scheduled})}")
    print(f"reports: {report_dir}")


if __name__ == "__main__":
    main()
