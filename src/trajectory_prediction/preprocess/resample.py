"""Per-aircraft one-hertz aggregation and short-gap interpolation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .deduplicate import DeduplicatedPoint


@dataclass(slots=True)
class ResampledPoint:
    timestamp: float
    longitude: float
    latitude: float
    geo_altitude: float
    baro_altitude: float | None
    num_measurements: int
    source_record_count: int
    is_interpolated: bool
    is_speed_suspicious: bool = False


@dataclass(frozen=True, slots=True)
class ResampledRun:
    trajectory_index: int
    points: list[ResampledPoint]


def _median_optional(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(statistics.median(valid)) if valid else None


def _aggregate_bin(timestamp: int, points: list[DeduplicatedPoint]) -> ResampledPoint:
    return ResampledPoint(
        timestamp=float(timestamp),
        longitude=float(statistics.median(point.longitude for point in points)),
        latitude=float(statistics.median(point.latitude for point in points)),
        geo_altitude=float(statistics.median(point.geo_altitude for point in points)),
        baro_altitude=_median_optional([point.baro_altitude for point in points]),
        num_measurements=max(point.num_measurements for point in points),
        source_record_count=sum(point.source_record_count for point in points),
        is_interpolated=False,
    )


def _interpolate(left: ResampledPoint, right: ResampledPoint, timestamp: int) -> ResampledPoint:
    ratio = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
    if left.baro_altitude is None or right.baro_altitude is None:
        baro_altitude = None
    else:
        baro_altitude = left.baro_altitude + (
            right.baro_altitude - left.baro_altitude
        ) * ratio
    return ResampledPoint(
        timestamp=float(timestamp),
        longitude=left.longitude + (right.longitude - left.longitude) * ratio,
        latitude=left.latitude + (right.latitude - left.latitude) * ratio,
        geo_altitude=left.geo_altitude
        + (right.geo_altitude - left.geo_altitude) * ratio,
        baro_altitude=baro_altitude,
        num_measurements=0,
        source_record_count=0,
        is_interpolated=True,
    )


def resample_aircraft(
    raw_points: list[DeduplicatedPoint],
    sampling_interval_seconds: int,
    macro_gap_seconds: float,
    max_interpolation_missing_steps: int,
) -> list[ResampledRun]:
    """Aggregate to a fixed grid and return model-safe continuous runs."""

    if not raw_points:
        return []
    bins: dict[int, list[DeduplicatedPoint]] = {}
    for point in raw_points:
        sample_time = (
            math.floor(point.timestamp / sampling_interval_seconds)
            * sampling_interval_seconds
        )
        bins.setdefault(sample_time, []).append(point)
    observed = [
        _aggregate_bin(timestamp, bins[timestamp]) for timestamp in sorted(bins)
    ]
    if not observed:
        return []

    runs: list[ResampledRun] = []
    trajectory_index = 1
    current: list[ResampledPoint] = [observed[0]]
    previous = observed[0]
    for point in observed[1:]:
        gap_seconds = point.timestamp - previous.timestamp
        missing_steps = int(round(gap_seconds / sampling_interval_seconds)) - 1
        if gap_seconds > macro_gap_seconds:
            runs.append(ResampledRun(trajectory_index, current))
            trajectory_index += 1
            current = [point]
        elif missing_steps > max_interpolation_missing_steps:
            runs.append(ResampledRun(trajectory_index, current))
            current = [point]
        else:
            for step in range(1, missing_steps + 1):
                timestamp = int(previous.timestamp) + step * sampling_interval_seconds
                current.append(_interpolate(previous, point, timestamp))
            current.append(point)
        previous = point
    runs.append(ResampledRun(trajectory_index, current))
    return runs
