"""Read-only access to aircraft trajectories stored as Parquet."""

from __future__ import annotations

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


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
