"""Split resampled runs at physically implausible horizontal jumps."""

from __future__ import annotations

from dataclasses import dataclass

from .quality import haversine_m
from .resample import ResampledPoint, ResampledRun


@dataclass(frozen=True, slots=True)
class SegmentResult:
    trajectory_index: int
    points: list[ResampledPoint]


@dataclass(frozen=True, slots=True)
class SegmentStats:
    suspicious_transitions: int
    strong_anomaly_breaks: int
    short_segments_discarded: int
    short_segment_points_discarded: int


def split_runs_on_speed(
    runs: list[ResampledRun],
    sampling_interval_seconds: int,
    suspicious_speed_mps: float,
    strong_anomaly_speed_mps: float,
    minimum_segment_points: int,
    minimum_segment_duration_seconds: float = 0.0,
) -> tuple[list[SegmentResult], SegmentStats]:
    segments: list[SegmentResult] = []
    suspicious = 0
    strong = 0
    short_segments = 0
    short_points = 0

    def keep_or_count(trajectory_index: int, points: list[ResampledPoint]) -> None:
        nonlocal short_segments, short_points
        duration_seconds = (
            points[-1].timestamp - points[0].timestamp if len(points) >= 2 else 0.0
        )
        if (
            len(points) >= minimum_segment_points
            and duration_seconds >= minimum_segment_duration_seconds
        ):
            segments.append(SegmentResult(trajectory_index, points))
        else:
            short_segments += 1
            short_points += len(points)

    for run in runs:
        if not run.points:
            continue
        current = [run.points[0]]
        previous = run.points[0]
        for point in run.points[1:]:
            dt = point.timestamp - previous.timestamp
            if dt <= 0:
                keep_or_count(run.trajectory_index, current)
                current = [point]
                strong += 1
                previous = point
                continue
            speed = haversine_m(
                previous.longitude,
                previous.latitude,
                point.longitude,
                point.latitude,
            ) / dt
            if speed > suspicious_speed_mps:
                suspicious += 1
                point.is_speed_suspicious = True
            if speed > strong_anomaly_speed_mps:
                strong += 1
                keep_or_count(run.trajectory_index, current)
                current = [point]
            else:
                current.append(point)
            previous = point
        keep_or_count(run.trajectory_index, current)

    return segments, SegmentStats(suspicious, strong, short_segments, short_points)
