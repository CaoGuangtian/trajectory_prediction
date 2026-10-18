"""Read-only access to aircraft trajectories stored as Parquet."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class AircraftSummary:
    aircraft: int
    point_count: int
    spatial_point_count: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class TrackData:
    aircraft: int
    time: tuple[float, ...]
    longitude: tuple[float | None, ...]
    latitude: tuple[float | None, ...]
    geo_altitude: tuple[float | None, ...]
    baro_altitude: tuple[float | None, ...]

    @property
    def point_count(self) -> int:
        return len(self.time)

    def between(self, start_time: float, end_time: float) -> "TrackData":
        """Return the observations inside an inclusive time window."""
        left = bisect_left(self.time, start_time)
        right = bisect_right(self.time, end_time)
        return TrackData(
            aircraft=self.aircraft,
            time=self.time[left:right],
            longitude=self.longitude[left:right],
            latitude=self.latitude[left:right],
            geo_altitude=self.geo_altitude[left:right],
            baro_altitude=self.baro_altitude[left:right],
        )


@dataclass(frozen=True)
class SamplingDistributionSummary:
    total_count: int
    detail_count: int
    detail_mean: float
    detail_standard_deviation: float
    detail_limit: float
    detail_counts: tuple[int, ...]
    maximum_interval: float
    gap_counts: tuple[int, ...]


class TrajectoryStore:
    """Query a trajectory Parquet file without loading it in full."""

    REQUIRED_COLUMNS = {
        "aircraft",
        "timeAtServer",
        "longitude",
        "latitude",
        "geoAltitude",
        "baroAltitude",
    }

    def __init__(self, parquet_path: str | Path) -> None:
        self.parquet_path = Path(parquet_path).resolve()
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"Trajectory data not found: {self.parquet_path}")
        self._validate_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:", read_only=False)

    def _validate_schema(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(self.parquet_path)],
            ).fetchall()
        columns = {row[0] for row in rows}
        missing = sorted(self.REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "Trajectory data is missing required columns: " + ", ".join(missing)
            )

    @lru_cache(maxsize=1)
    def aircraft_summaries(self) -> tuple[AircraftSummary, ...]:
        query = """
            SELECT
                aircraft,
                count(*) AS point_count,
                count(*) FILTER (
                    WHERE longitude IS NOT NULL AND latitude IS NOT NULL
                ) AS spatial_point_count,
                min(timeAtServer) AS start_time,
                max(timeAtServer) AS end_time
            FROM read_parquet(?)
            WHERE aircraft IS NOT NULL AND timeAtServer IS NOT NULL
            GROUP BY aircraft
            ORDER BY aircraft
        """
        with self._connect() as connection:
            rows = connection.execute(query, [str(self.parquet_path)]).fetchall()
        return tuple(
            AircraftSummary(
                aircraft=int(row[0]),
                point_count=int(row[1]),
                spatial_point_count=int(row[2]),
                start_time=float(row[3]),
                end_time=float(row[4]),
            )
            for row in rows
        )

    @lru_cache(maxsize=8)
    def load_track(self, aircraft: int) -> TrackData:
        query = """
            SELECT
                timeAtServer,
                longitude,
                latitude,
                geoAltitude,
                baroAltitude
            FROM read_parquet(?)
            WHERE aircraft = ? AND timeAtServer IS NOT NULL
            ORDER BY timeAtServer
        """
        with self._connect() as connection:
            rows = connection.execute(
                query,
                [str(self.parquet_path), int(aircraft)],
            ).fetchall()

        return TrackData(
            aircraft=int(aircraft),
            time=tuple(float(row[0]) for row in rows),
            longitude=tuple(_optional_float(row[1]) for row in rows),
            latitude=tuple(_optional_float(row[2]) for row in rows),
            geo_altitude=tuple(_optional_float(row[3]) for row in rows),
            baro_altitude=tuple(_optional_float(row[4]) for row in rows),
        )

    @lru_cache(maxsize=1)
    def global_sampling_distribution(
        self,
        detail_limit: float = 2.0,
        detail_bin_count: int = 40,
        gap_bin_count: int = 24,
    ) -> SamplingDistributionSummary:
        """Aggregate adjacent time differences without loading all rows into Python."""
        query = """
            WITH
            parameters AS (
                SELECT
                    ?::DOUBLE AS detail_limit,
                    ?::INTEGER AS detail_bin_count,
                    ?::INTEGER AS gap_bin_count
            ),
            ordered_observations AS (
                SELECT
                    aircraft,
                    timeAtServer,
                    lag(timeAtServer) OVER (
                        PARTITION BY aircraft
                        ORDER BY timeAtServer
                    ) AS previous_time
                FROM read_parquet(?)
                WHERE aircraft IS NOT NULL AND timeAtServer IS NOT NULL
            ),
            intervals AS (
                SELECT timeAtServer - previous_time AS sampling_interval
                FROM ordered_observations
                WHERE previous_time IS NOT NULL AND timeAtServer >= previous_time
            ),
            statistics AS (
                SELECT
                    count(*) AS total_count,
                    count(*) FILTER (
                        WHERE sampling_interval <= parameters.detail_limit
                    ) AS detail_count,
                    avg(sampling_interval) FILTER (
                        WHERE sampling_interval <= parameters.detail_limit
                    ) AS detail_mean,
                    stddev_pop(sampling_interval) FILTER (
                        WHERE sampling_interval <= parameters.detail_limit
                    ) AS detail_standard_deviation,
                    max(sampling_interval) AS maximum_interval
                FROM intervals
                CROSS JOIN parameters
                GROUP BY parameters.detail_limit
            ),
            binned AS (
                SELECT
                    CASE
                        WHEN sampling_interval <= parameters.detail_limit
                            THEN 'detail'
                        ELSE 'gap'
                    END AS region,
                    CASE
                        WHEN sampling_interval <= parameters.detail_limit THEN least(
                            floor(
                                sampling_interval /
                                (parameters.detail_limit / parameters.detail_bin_count)
                            )::INTEGER,
                            parameters.detail_bin_count - 1
                        )
                        ELSE least(
                            floor(
                                (sampling_interval - parameters.detail_limit) /
                                greatest(
                                    (statistics.maximum_interval - parameters.detail_limit) /
                                    parameters.gap_bin_count,
                                    0.001
                                )
                            )::INTEGER,
                            parameters.gap_bin_count - 1
                        )
                    END AS bin_index,
                    statistics.total_count,
                    statistics.detail_count,
                    statistics.detail_mean,
                    statistics.detail_standard_deviation,
                    statistics.maximum_interval
                FROM intervals
                CROSS JOIN parameters
                CROSS JOIN statistics
            )
            SELECT
                region,
                bin_index,
                count(*) AS bin_count,
                max(total_count) AS total_count,
                max(detail_count) AS detail_count,
                max(detail_mean) AS detail_mean,
                max(detail_standard_deviation) AS detail_standard_deviation,
                max(maximum_interval) AS maximum_interval
            FROM binned
            GROUP BY region, bin_index
            ORDER BY region, bin_index
        """
        with self._connect() as connection:
            rows = connection.execute(
                query,
                [
                    float(detail_limit),
                    int(detail_bin_count),
                    int(gap_bin_count),
                    str(self.parquet_path),
                ],
            ).fetchall()

        if not rows:
            return SamplingDistributionSummary(
                total_count=0,
                detail_count=0,
                detail_mean=0.0,
                detail_standard_deviation=0.0,
                detail_limit=float(detail_limit),
                detail_counts=(0,) * int(detail_bin_count),
                maximum_interval=float(detail_limit),
                gap_counts=(0,) * int(gap_bin_count),
            )

        detail_counts = [0] * int(detail_bin_count)
        gap_counts = [0] * int(gap_bin_count)
        for region, bin_index, bin_count, *_ in rows:
            target = detail_counts if region == "detail" else gap_counts
            target[int(bin_index)] = int(bin_count)

        first_row = rows[0]
        return SamplingDistributionSummary(
            total_count=int(first_row[3]),
            detail_count=int(first_row[4]),
            detail_mean=float(first_row[5] or 0.0),
            detail_standard_deviation=float(first_row[6] or 0.0),
            detail_limit=float(detail_limit),
            detail_counts=tuple(detail_counts),
            maximum_interval=float(first_row[7] or detail_limit),
            gap_counts=tuple(gap_counts),
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
