"""Streaming duplicate-time consolidation backed by DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import duckdb


@dataclass(frozen=True, slots=True)
class DeduplicatedPoint:
    aircraft: int
    timestamp: float
    longitude: float
    latitude: float
    geo_altitude: float
    baro_altitude: float | None
    num_measurements: int
    source_record_count: int


@dataclass(frozen=True, slots=True)
class SourceStats:
    source_rows: int
    valid_position_rows: int
    invalid_position_rows: int
    deduplicated_timestamps: int
    duplicate_rows_removed: int
    conflict_timestamps_removed: int
    conflict_source_rows_removed: int


VALID_POSITION_SQL = """
    latitude IS NOT NULL
    AND longitude IS NOT NULL
    AND geoAltitude IS NOT NULL
    AND isfinite(latitude)
    AND isfinite(longitude)
    AND isfinite(geoAltitude)
    AND latitude BETWEEN -90 AND 90
    AND longitude BETWEEN -180 AND 180
"""


def _grouped_sql() -> str:
    return f"""
        WITH valid AS (
            SELECT
                aircraft,
                timeAtServer,
                longitude,
                latitude,
                geoAltitude,
                CASE
                    WHEN baroAltitude IS NOT NULL AND isfinite(baroAltitude)
                    THEN baroAltitude
                    ELSE NULL
                END AS baroAltitude,
                numMeasurements
            FROM read_parquet(?)
            WHERE {VALID_POSITION_SQL}
                AND timeAtServer IS NOT NULL
                AND isfinite(timeAtServer)
        ),
        grouped AS (
            SELECT
                aircraft,
                timeAtServer,
                median(longitude) AS longitude,
                median(latitude) AS latitude,
                median(geoAltitude) AS geoAltitude,
                median(baroAltitude) AS baroAltitude,
                max(numMeasurements) AS numMeasurements,
                count(*)::INTEGER AS source_record_count,
                min(longitude) AS min_lon,
                max(longitude) AS max_lon,
                min(latitude) AS min_lat,
                max(latitude) AS max_lat
            FROM valid
            GROUP BY aircraft, timeAtServer
        ),
        measured AS (
            SELECT
                *,
                sqrt(
                    pow((max_lat - min_lat) * 111320.0, 2)
                    + pow(
                        (max_lon - min_lon) * 111320.0
                        * greatest(0.1, cos(radians(latitude))),
                        2
                    )
                ) AS coordinate_spread_m
            FROM grouped
        )
    """


def collect_source_stats(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    conflict_threshold_m: float,
) -> SourceStats:
    source_rows, valid_rows = connection.execute(
        f"""
        SELECT
            count(*)::BIGINT,
            count(*) FILTER (WHERE {VALID_POSITION_SQL})::BIGINT
        FROM read_parquet(?)
        """,
        [str(parquet_path)],
    ).fetchone()
    grouped = connection.execute(
        _grouped_sql()
        + """
        SELECT
            count(*)::BIGINT,
            coalesce(sum(source_record_count - 1), 0)::BIGINT,
            count(*) FILTER (WHERE coordinate_spread_m > ?)::BIGINT,
            coalesce(
                sum(source_record_count) FILTER (WHERE coordinate_spread_m > ?),
                0
            )::BIGINT
        FROM measured
        """,
        [str(parquet_path), conflict_threshold_m, conflict_threshold_m],
    ).fetchone()
    return SourceStats(
        source_rows=int(source_rows),
        valid_position_rows=int(valid_rows),
        invalid_position_rows=int(source_rows - valid_rows),
        deduplicated_timestamps=int(grouped[0] - grouped[2]),
        duplicate_rows_removed=int(grouped[1]),
        conflict_timestamps_removed=int(grouped[2]),
        conflict_source_rows_removed=int(grouped[3]),
    )


def iter_deduplicated_points(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    conflict_threshold_m: float,
    batch_size: int = 100_000,
) -> Iterator[DeduplicatedPoint]:
    reader = connection.execute(
        _grouped_sql()
        + """
        SELECT
            aircraft,
            timeAtServer,
            longitude,
            latitude,
            geoAltitude,
            baroAltitude,
            numMeasurements,
            source_record_count
        FROM measured
        WHERE coordinate_spread_m <= ?
        ORDER BY aircraft, timeAtServer
        """,
        [str(parquet_path), conflict_threshold_m],
    ).fetch_record_batch(batch_size)
    for batch in reader:
        columns = batch.to_pydict()
        for values in zip(
            columns["aircraft"],
            columns["timeAtServer"],
            columns["longitude"],
            columns["latitude"],
            columns["geoAltitude"],
            columns["baroAltitude"],
            columns["numMeasurements"],
            columns["source_record_count"],
        ):
            yield DeduplicatedPoint(
                aircraft=int(values[0]),
                timestamp=float(values[1]),
                longitude=float(values[2]),
                latitude=float(values[3]),
                geo_altitude=float(values[4]),
                baro_altitude=None if values[5] is None else float(values[5]),
                num_measurements=int(values[6]),
                source_record_count=int(values[7]),
            )
