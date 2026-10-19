"""Constant-velocity linear-extrapolation baseline evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .position_hold import (
    EARTH_MEAN_RADIUS_M,
    _create_source_view,
    _horizon_metrics_sql,
    _segment_metrics_sql,
    _source_manifest_entry,
    _write_csv_atomic,
    _write_json_atomic,
    _write_parquet_atomic,
)


@dataclass(frozen=True, slots=True)
class ConstantVelocitySource:
    subset_id: int
    parquet_path: Path
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ConstantVelocityConfig:
    version: str
    dataset_root: Path
    subsets: tuple[int, ...]
    split: str
    history_seconds: int
    prediction_horizons_seconds: tuple[int, ...]
    origin_stride_seconds: int
    exclude_interpolated_history: bool
    exclude_suspicious_history: bool
    minimum_history_points: int
    minimum_history_span_seconds: int
    exclude_interpolated_origins: bool
    exclude_suspicious_origins: bool
    exclude_interpolated_targets: bool
    exclude_suspicious_targets: bool
    output_root: Path
    duckdb_memory_limit: str
    duckdb_threads: int
    source_path: Path
    config_hash: str

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_type": "constant_velocity",
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "split": self.split,
            "history_seconds": self.history_seconds,
            "prediction_horizons_seconds": list(
                self.prediction_horizons_seconds
            ),
            "origin_stride_seconds": self.origin_stride_seconds,
            "velocity_estimator": "least_squares_slope",
            "position_anchor": "forecast_origin",
            "coordinate_system": "local_tangent_plane_approximation",
            "exclude_interpolated_history": self.exclude_interpolated_history,
            "exclude_suspicious_history": self.exclude_suspicious_history,
            "minimum_history_points": self.minimum_history_points,
            "minimum_history_span_seconds": self.minimum_history_span_seconds,
            "insufficient_history_fallback": "position_hold",
            "exclude_interpolated_origins": self.exclude_interpolated_origins,
            "exclude_suspicious_origins": self.exclude_suspicious_origins,
            "exclude_interpolated_targets": self.exclude_interpolated_targets,
            "exclude_suspicious_targets": self.exclude_suspicious_targets,
            "output_root": self.output_root.as_posix(),
            "duckdb_memory_limit": self.duckdb_memory_limit,
            "duckdb_threads": self.duckdb_threads,
            "config_hash": self.config_hash,
        }


def load_constant_velocity_config(path: Path) -> ConstantVelocityConfig:
    """Load the JSON-compatible YAML configuration."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Constant-velocity configuration must be an object.")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config = ConstantVelocityConfig(
        version=str(_required(payload, "version")),
        dataset_root=Path(_required(payload, "dataset_root")),
        subsets=tuple(int(value) for value in _required(payload, "subsets")),
        split=str(_required(payload, "split")),
        history_seconds=int(_required(payload, "history_seconds")),
        prediction_horizons_seconds=tuple(
            int(value)
            for value in _required(payload, "prediction_horizons_seconds")
        ),
        origin_stride_seconds=int(_required(payload, "origin_stride_seconds")),
        exclude_interpolated_history=bool(
            _required(payload, "exclude_interpolated_history")
        ),
        exclude_suspicious_history=bool(
            _required(payload, "exclude_suspicious_history")
        ),
        minimum_history_points=int(_required(payload, "minimum_history_points")),
        minimum_history_span_seconds=int(
            _required(payload, "minimum_history_span_seconds")
        ),
        exclude_interpolated_origins=bool(
            _required(payload, "exclude_interpolated_origins")
        ),
        exclude_suspicious_origins=bool(
            _required(payload, "exclude_suspicious_origins")
        ),
        exclude_interpolated_targets=bool(
            _required(payload, "exclude_interpolated_targets")
        ),
        exclude_suspicious_targets=bool(
            _required(payload, "exclude_suspicious_targets")
        ),
        output_root=Path(_required(payload, "output_root")),
        duckdb_memory_limit=str(_required(payload, "duckdb_memory_limit")),
        duckdb_threads=int(_required(payload, "duckdb_threads")),
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    _validate_config(config)
    return config


def evaluate_constant_velocity(
    sources: Iterable[ConstantVelocitySource],
    config: ConstantVelocityConfig,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate causal constant-velocity extrapolation and write aggregates."""

    selected_sources = tuple(sorted(sources, key=lambda item: item.subset_id))
    if not selected_sources:
        raise ValueError("At least one standardized subset is required.")
    subset_ids = [source.subset_id for source in selected_sources]
    if len(set(subset_ids)) != len(subset_ids):
        raise ValueError("Constant-velocity sources contain duplicate subset IDs.")
    for source in selected_sources:
        if not source.parquet_path.is_file():
            raise FileNotFoundError(
                f"Standardized trajectory artifact does not exist: {source.parquet_path}"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.yaml"
    horizon_path = output_directory / "metrics_by_horizon.csv"
    segment_path = output_directory / "metrics_by_segment.parquet"
    existing = [path for path in (manifest_path, horizon_path, segment_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Constant-velocity output already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    temporary_directory = tempfile.TemporaryDirectory(
        prefix=".constant_velocity_duckdb_",
        dir=output_directory,
    )
    connection = duckdb.connect()
    try:
        escaped_temporary = temporary_directory.name.replace("'", "''")
        escaped_memory = config.duckdb_memory_limit.replace("'", "''")
        connection.execute(f"SET temp_directory = '{escaped_temporary}'")
        connection.execute(f"SET memory_limit = '{escaped_memory}'")
        connection.execute(f"SET threads = {config.duckdb_threads}")
        connection.execute("SET preserve_insertion_order = false")
        _create_source_view(connection, selected_sources)
        _create_model_state_tables(connection, config, Path(temporary_directory.name))
        total_origins, fallback_origins = _origin_counts(connection, config)

        segment_tables: list[pa.Table] = []
        horizon_rows: list[tuple[object, ...]] = []
        horizon_columns: list[str] | None = None
        for horizon_s in config.prediction_horizons_seconds:
            _create_evaluation_table(connection, config, horizon_s)
            segment_part_path = (
                Path(temporary_directory.name) / f"segment_metrics_{horizon_s}.parquet"
            )
            escaped_segment_part_path = str(segment_part_path.resolve()).replace(
                "'", "''"
            )
            connection.execute(
                f"""
                COPY ({_segment_metrics_sql()})
                TO '{escaped_segment_part_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            segment_tables.append(pq.read_table(segment_part_path))
            horizon_cursor = connection.execute(_horizon_metrics_sql())
            columns = [description[0] for description in horizon_cursor.description]
            if horizon_columns is None:
                horizon_columns = columns
            elif horizon_columns != columns:
                raise RuntimeError("Inconsistent horizon-metric schema across horizons.")
            horizon_rows.extend(horizon_cursor.fetchall())
            connection.execute("DROP TABLE evaluation_rows")

        if not segment_tables or horizon_columns is None:
            raise RuntimeError("Constant-velocity evaluation produced no horizons.")
        segment_table = pa.concat_tables(segment_tables)

        _write_parquet_atomic(segment_path, segment_table, overwrite)
        _write_csv_atomic(horizon_path, horizon_columns, horizon_rows, overwrite)
    finally:
        connection.close()
        temporary_directory.cleanup()

    source_entries = [_source_manifest_entry(source) for source in selected_sources]
    horizon_records = [
        dict(zip(horizon_columns, row, strict=True)) for row in horizon_rows
    ]
    candidate_predictions = sum(
        int(record["candidate_origins"]) for record in horizon_records
    )
    evaluated_targets = sum(
        int(record["evaluated_targets"]) for record in horizon_records
    )
    evaluated_segments = len(
        {
            (int(subset), int(segment))
            for subset, segment in zip(
                segment_table["subset_id"].to_pylist(),
                segment_table["segment_id"].to_pylist(),
                strict=True,
            )
        }
    )
    manifest: dict[str, object] = {
        "artifact": "constant_velocity_baseline_evaluation",
        "model_type": "constant_velocity",
        "display_name": "Constant Velocity",
        "evaluation_protocol": "available targets by horizon",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.as_manifest_dict(),
        "sources": source_entries,
        "outputs": {
            "metrics_by_horizon": horizon_path.as_posix(),
            "metrics_by_segment": segment_path.as_posix(),
        },
        "statistics": {
            "selected_subsets": subset_ids,
            "evaluated_segments": evaluated_segments,
            "segment_horizon_rows": segment_table.num_rows,
            "unique_prediction_origins": total_origins,
            "velocity_fallback_origins": fallback_origins,
            "velocity_fallback_rate": (
                fallback_origins / total_origins if total_origins else 0.0
            ),
            "candidate_predictions": candidate_predictions,
            "evaluated_targets": evaluated_targets,
            "overall_target_coverage": (
                evaluated_targets / candidate_predictions
                if candidate_predictions
                else 0.0
            ),
        },
        "notes": [
            "Velocity is estimated causally by least-squares slopes over the trailing history window.",
            "Predictions are anchored at the forecast-origin position, not at the fitted intercept.",
            "Insufficient valid history falls back to zero velocity without dropping the origin.",
            "Horizontal error uses the WGS84 longitude/latitude great-circle distance in metres.",
            "Vertical error uses geometric altitude; barometric-altitude error is auxiliary.",
            "Prediction windows never cross segment_id boundaries.",
            "No per-origin prediction rows or model-input windows are persisted.",
            "JSON syntax is used because JSON is a valid YAML 1.2 subset.",
        ],
    }
    _write_json_atomic(manifest_path, manifest, overwrite)
    return manifest


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing constant-velocity configuration key: {key}")
    return mapping[key]


def _validate_config(config: ConstantVelocityConfig) -> None:
    if not config.version.strip():
        raise ValueError("version must not be empty.")
    if not config.subsets or any(value < 1 for value in config.subsets):
        raise ValueError("subsets must contain positive integers.")
    if config.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if config.history_seconds < 1:
        raise ValueError("history_seconds must be positive.")
    horizons = config.prediction_horizons_seconds
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("prediction_horizons_seconds must contain positive integers.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError(
            "prediction_horizons_seconds must be unique and strictly increasing."
        )
    if config.origin_stride_seconds < 1:
        raise ValueError("origin_stride_seconds must be positive.")
    if config.minimum_history_points < 2:
        raise ValueError("minimum_history_points must be at least two.")
    if not 1 <= config.minimum_history_span_seconds <= config.history_seconds:
        raise ValueError(
            "minimum_history_span_seconds must be between one and history_seconds."
        )
    if not config.duckdb_memory_limit.strip():
        raise ValueError("duckdb_memory_limit must not be empty.")
    if config.duckdb_threads < 1:
        raise ValueError("duckdb_threads must be positive.")


def _create_evaluation_view(
    connection: duckdb.DuckDBPyConnection,
    config: ConstantVelocityConfig,
) -> None:
    horizons = ", ".join(
        f"({int(value)})" for value in config.prediction_horizons_seconds
    )
    split = config.split.replace("'", "''")
    history_conditions = ["longitude IS NOT NULL", "latitude IS NOT NULL"]
    if config.exclude_interpolated_history:
        history_conditions.append("NOT is_interpolated")
    if config.exclude_suspicious_history:
        history_conditions.append("NOT is_speed_suspicious")
    valid_history = " AND ".join(history_conditions)
    valid_velocity = (
        f"velocity_history_points >= {config.minimum_history_points} "
        f"AND velocity_history_span_s >= {config.minimum_history_span_seconds}"
    )
    origin_conditions = [
        f"o.split = '{split}'",
        (
            "CAST(round(o.timeAtServer - b.segment_start_time) AS BIGINT) "
            f"% {config.origin_stride_seconds} = 0"
        ),
        f"o.timeAtServer >= b.segment_start_time + {config.history_seconds}",
        "o.timeAtServer + h.horizon_s <= b.segment_end_time",
    ]
    if config.exclude_interpolated_origins:
        origin_conditions.append("NOT o.is_interpolated")
    if config.exclude_suspicious_origins:
        origin_conditions.append("NOT o.is_speed_suspicious")
    target_conditions = ["t.timeAtServer IS NOT NULL"]
    if config.exclude_interpolated_targets:
        target_conditions.append("NOT t.is_interpolated")
    if config.exclude_suspicious_targets:
        target_conditions.append("NOT t.is_speed_suspicious")
    target_valid = " AND ".join(target_conditions)
    horizontal_error = f"""
        2.0 * {EARTH_MEAN_RADIUS_M} * asin(
            sqrt(
                least(
                    1.0,
                    pow(sin(radians(t.latitude - c.predicted_latitude) / 2.0), 2)
                    + cos(radians(c.predicted_latitude)) * cos(radians(t.latitude))
                    * pow(sin(radians(t.longitude - c.predicted_longitude) / 2.0), 2)
                )
            )
        )
    """
    connection.execute(
        f"""
        CREATE TEMP VIEW evaluation_rows AS
        WITH segment_bounds AS (
            SELECT
                subset_id,
                aircraft,
                trajectory_id,
                segment_id,
                any_value(trusted) AS trusted,
                any_value(split) AS split,
                min(timeAtServer) AS segment_start_time,
                max(timeAtServer) AS segment_end_time,
                count(*) AS segment_points,
                count(*) FILTER (WHERE NOT is_interpolated) AS segment_observed_points,
                count(*) FILTER (WHERE is_interpolated) AS segment_interpolated_points
            FROM source_points
            WHERE split = '{split}'
            GROUP BY subset_id, aircraft, trajectory_id, segment_id
        ),
        history_estimates AS (
            SELECT
                *,
                count(*) FILTER (WHERE {valid_history}) OVER history_window
                    AS velocity_history_points,
                max(timeAtServer) FILTER (WHERE {valid_history}) OVER history_window
                    - min(timeAtServer) FILTER (WHERE {valid_history}) OVER history_window
                    AS velocity_history_span_s,
                regr_slope(longitude, timeAtServer)
                    FILTER (WHERE {valid_history}) OVER history_window
                    AS longitude_rate_deg_s,
                regr_slope(latitude, timeAtServer)
                    FILTER (WHERE {valid_history}) OVER history_window
                    AS latitude_rate_deg_s,
                regr_slope(geoAltitude, timeAtServer)
                    FILTER (WHERE {valid_history} AND geoAltitude IS NOT NULL)
                    OVER history_window AS geo_altitude_rate_m_s,
                regr_slope(baroAltitude, timeAtServer)
                    FILTER (WHERE {valid_history} AND baroAltitude IS NOT NULL)
                    OVER history_window AS baro_altitude_rate_m_s
            FROM source_points
            WHERE split = '{split}'
            WINDOW history_window AS (
                PARTITION BY subset_id, aircraft, trajectory_id, segment_id
                ORDER BY timeAtServer
                RANGE BETWEEN {config.history_seconds} PRECEDING AND CURRENT ROW
            )
        ),
        origin_states AS (
            SELECT
                *,
                NOT ({valid_velocity}) AS velocity_fallback,
                CASE WHEN {valid_velocity}
                    THEN coalesce(longitude_rate_deg_s, 0.0) ELSE 0.0
                END AS fitted_longitude_rate_deg_s,
                CASE WHEN {valid_velocity}
                    THEN coalesce(latitude_rate_deg_s, 0.0) ELSE 0.0
                END AS fitted_latitude_rate_deg_s,
                CASE WHEN {valid_velocity}
                    THEN coalesce(geo_altitude_rate_m_s, 0.0) ELSE 0.0
                END AS fitted_geo_altitude_rate_m_s,
                CASE WHEN {valid_velocity}
                    THEN coalesce(baro_altitude_rate_m_s, 0.0) ELSE 0.0
                END AS fitted_baro_altitude_rate_m_s
            FROM history_estimates
        ),
        horizons(horizon_s) AS (VALUES {horizons}),
        candidates AS (
            SELECT
                o.subset_id,
                o.aircraft,
                o.trajectory_id,
                o.segment_id,
                b.trusted,
                b.split,
                b.segment_start_time,
                b.segment_end_time,
                b.segment_end_time - b.segment_start_time AS segment_duration_s,
                b.segment_points,
                b.segment_observed_points,
                b.segment_interpolated_points,
                o.timeAtServer AS origin_time,
                o.velocity_history_points,
                o.velocity_history_span_s,
                o.velocity_fallback,
                h.horizon_s,
                o.longitude + o.fitted_longitude_rate_deg_s * h.horizon_s
                    AS predicted_longitude,
                o.latitude + o.fitted_latitude_rate_deg_s * h.horizon_s
                    AS predicted_latitude,
                o.geoAltitude + o.fitted_geo_altitude_rate_m_s * h.horizon_s
                    AS predicted_geo_altitude,
                CASE WHEN o.baroAltitude IS NOT NULL
                    THEN o.baroAltitude
                        + o.fitted_baro_altitude_rate_m_s * h.horizon_s
                END AS predicted_baro_altitude
            FROM origin_states AS o
            JOIN segment_bounds AS b
              USING (subset_id, aircraft, trajectory_id, segment_id)
            CROSS JOIN horizons AS h
            WHERE {' AND '.join(origin_conditions)}
        )
        SELECT
            c.*,
            ({target_valid}) AS target_is_evaluated,
            CASE WHEN {target_valid} THEN {horizontal_error} END AS horizontal_error_m,
            CASE WHEN {target_valid}
                THEN abs(t.geoAltitude - c.predicted_geo_altitude)
            END AS vertical_error_m,
            CASE WHEN {target_valid} THEN sqrt(
                pow({horizontal_error}, 2)
                + pow(t.geoAltitude - c.predicted_geo_altitude, 2)
            ) END AS error_3d_m,
            CASE WHEN {target_valid}
                AND c.predicted_baro_altitude IS NOT NULL
                AND t.baroAltitude IS NOT NULL
                THEN abs(t.baroAltitude - c.predicted_baro_altitude)
            END AS barometric_vertical_error_m
        FROM candidates AS c
        LEFT JOIN source_points AS t
          ON t.subset_id = c.subset_id
         AND t.aircraft = c.aircraft
         AND t.trajectory_id = c.trajectory_id
         AND t.segment_id = c.segment_id
         AND t.timeAtServer = c.origin_time + c.horizon_s
        """
    )


def _create_model_state_tables(
    connection: duckdb.DuckDBPyConnection,
    config: ConstantVelocityConfig,
    temporary_directory: Path,
) -> None:
    """Materialize one compact velocity state per standardized trajectory point."""

    split = config.split.replace("'", "''")
    valid_history = _valid_history_sql(config)
    valid_velocity = _valid_velocity_sql(config)
    origin_state_path = (temporary_directory / "origin_states.parquet").resolve()
    escaped_origin_state_path = str(origin_state_path).replace("'", "''")
    connection.execute(
        f"""
        CREATE TEMP TABLE segment_bounds AS
        SELECT
            subset_id,
            aircraft,
            trajectory_id,
            segment_id,
            any_value(trusted) AS trusted,
            any_value(split) AS split,
            min(timeAtServer) AS segment_start_time,
            max(timeAtServer) AS segment_end_time,
            count(*) AS segment_points,
            count(*) FILTER (WHERE NOT is_interpolated) AS segment_observed_points,
            count(*) FILTER (WHERE is_interpolated) AS segment_interpolated_points
        FROM source_points
        WHERE split = '{split}'
        GROUP BY subset_id, aircraft, trajectory_id, segment_id
        """
    )
    connection.execute(
        f"""
        COPY (
        WITH history_estimates AS (
            SELECT
                subset_id,
                aircraft,
                trajectory_id,
                segment_id,
                timeAtServer,
                longitude,
                latitude,
                geoAltitude,
                baroAltitude,
                split,
                is_interpolated,
                is_speed_suspicious,
                count(*) FILTER (WHERE {valid_history}) OVER history_window
                    AS velocity_history_points,
                max(timeAtServer) FILTER (WHERE {valid_history}) OVER history_window
                    - min(timeAtServer) FILTER (WHERE {valid_history}) OVER history_window
                    AS velocity_history_span_s,
                regr_slope(longitude, timeAtServer)
                    FILTER (WHERE {valid_history}) OVER history_window
                    AS longitude_rate_deg_s,
                regr_slope(latitude, timeAtServer)
                    FILTER (WHERE {valid_history}) OVER history_window
                    AS latitude_rate_deg_s,
                regr_slope(geoAltitude, timeAtServer)
                    FILTER (WHERE {valid_history} AND geoAltitude IS NOT NULL)
                    OVER history_window AS geo_altitude_rate_m_s,
                regr_slope(baroAltitude, timeAtServer)
                    FILTER (WHERE {valid_history} AND baroAltitude IS NOT NULL)
                    OVER history_window AS baro_altitude_rate_m_s
            FROM source_points
            WHERE split = '{split}'
            WINDOW history_window AS (
                PARTITION BY subset_id, aircraft, trajectory_id, segment_id
                ORDER BY timeAtServer
                RANGE BETWEEN {config.history_seconds} PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            subset_id,
            aircraft,
            trajectory_id,
            segment_id,
            timeAtServer,
            longitude,
            latitude,
            geoAltitude,
            baroAltitude,
            split,
            is_interpolated,
            is_speed_suspicious,
            velocity_history_points,
            velocity_history_span_s,
            NOT ({valid_velocity}) AS velocity_fallback,
            CASE WHEN {valid_velocity}
                THEN coalesce(longitude_rate_deg_s, 0.0) ELSE 0.0
            END AS fitted_longitude_rate_deg_s,
            CASE WHEN {valid_velocity}
                THEN coalesce(latitude_rate_deg_s, 0.0) ELSE 0.0
            END AS fitted_latitude_rate_deg_s,
            CASE WHEN {valid_velocity}
                THEN coalesce(geo_altitude_rate_m_s, 0.0) ELSE 0.0
            END AS fitted_geo_altitude_rate_m_s,
            CASE WHEN {valid_velocity}
                THEN coalesce(baro_altitude_rate_m_s, 0.0) ELSE 0.0
            END AS fitted_baro_altitude_rate_m_s
        FROM history_estimates
        ) TO '{escaped_origin_state_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW origin_states AS
        SELECT * FROM read_parquet('{escaped_origin_state_path}')
        """
    )


def _origin_counts(
    connection: duckdb.DuckDBPyConnection,
    config: ConstantVelocityConfig,
) -> tuple[int, int]:
    conditions = _origin_conditions(config, min(config.prediction_horizons_seconds))
    row = connection.execute(
        f"""
        SELECT
            count(*)::BIGINT AS total_origins,
            count(*) FILTER (WHERE o.velocity_fallback)::BIGINT AS fallback_origins
        FROM origin_states AS o
        JOIN segment_bounds AS b
          USING (subset_id, aircraft, trajectory_id, segment_id)
        WHERE {' AND '.join(conditions)}
        """
    ).fetchone()
    return int(row[0]), int(row[1])


def _create_evaluation_table(
    connection: duckdb.DuckDBPyConnection,
    config: ConstantVelocityConfig,
    horizon_s: int,
) -> None:
    conditions = _origin_conditions(config, horizon_s)
    target_conditions = ["t.timeAtServer IS NOT NULL"]
    if config.exclude_interpolated_targets:
        target_conditions.append("NOT t.is_interpolated")
    if config.exclude_suspicious_targets:
        target_conditions.append("NOT t.is_speed_suspicious")
    target_valid = " AND ".join(target_conditions)
    horizontal_error = f"""
        2.0 * {EARTH_MEAN_RADIUS_M} * asin(
            sqrt(
                least(
                    1.0,
                    pow(sin(radians(t.latitude - c.predicted_latitude) / 2.0), 2)
                    + cos(radians(c.predicted_latitude)) * cos(radians(t.latitude))
                    * pow(sin(radians(t.longitude - c.predicted_longitude) / 2.0), 2)
                )
            )
        )
    """
    connection.execute(
        f"""
        CREATE TEMP TABLE evaluation_rows AS
        WITH candidates AS (
            SELECT
                o.subset_id,
                o.aircraft,
                o.trajectory_id,
                o.segment_id,
                b.trusted,
                b.split,
                b.segment_start_time,
                b.segment_end_time,
                b.segment_end_time - b.segment_start_time AS segment_duration_s,
                b.segment_points,
                b.segment_observed_points,
                b.segment_interpolated_points,
                {int(horizon_s)}::INTEGER AS horizon_s,
                o.timeAtServer AS origin_time,
                o.longitude + o.fitted_longitude_rate_deg_s * {int(horizon_s)}
                    AS predicted_longitude,
                o.latitude + o.fitted_latitude_rate_deg_s * {int(horizon_s)}
                    AS predicted_latitude,
                o.geoAltitude + o.fitted_geo_altitude_rate_m_s * {int(horizon_s)}
                    AS predicted_geo_altitude,
                CASE WHEN o.baroAltitude IS NOT NULL
                    THEN o.baroAltitude
                        + o.fitted_baro_altitude_rate_m_s * {int(horizon_s)}
                END AS predicted_baro_altitude
            FROM origin_states AS o
            JOIN segment_bounds AS b
              USING (subset_id, aircraft, trajectory_id, segment_id)
            WHERE {' AND '.join(conditions)}
        )
        SELECT
            c.subset_id,
            c.aircraft,
            c.trajectory_id,
            c.segment_id,
            c.trusted,
            c.split,
            c.segment_start_time,
            c.segment_end_time,
            c.segment_duration_s,
            c.segment_points,
            c.segment_observed_points,
            c.segment_interpolated_points,
            c.horizon_s,
            CASE WHEN {target_valid} THEN {horizontal_error} END AS horizontal_error_m,
            CASE WHEN {target_valid}
                THEN abs(t.geoAltitude - c.predicted_geo_altitude)
            END AS vertical_error_m,
            CASE WHEN {target_valid} THEN sqrt(
                pow({horizontal_error}, 2)
                + pow(t.geoAltitude - c.predicted_geo_altitude, 2)
            ) END AS error_3d_m,
            CASE WHEN {target_valid}
                AND c.predicted_baro_altitude IS NOT NULL
                AND t.baroAltitude IS NOT NULL
                THEN abs(t.baroAltitude - c.predicted_baro_altitude)
            END AS barometric_vertical_error_m
        FROM candidates AS c
        LEFT JOIN source_points AS t
          ON t.subset_id = c.subset_id
         AND t.aircraft = c.aircraft
         AND t.trajectory_id = c.trajectory_id
         AND t.segment_id = c.segment_id
         AND t.timeAtServer = c.origin_time + c.horizon_s
        """
    )


def _valid_history_sql(config: ConstantVelocityConfig) -> str:
    conditions = ["longitude IS NOT NULL", "latitude IS NOT NULL"]
    if config.exclude_interpolated_history:
        conditions.append("NOT is_interpolated")
    if config.exclude_suspicious_history:
        conditions.append("NOT is_speed_suspicious")
    return " AND ".join(conditions)


def _valid_velocity_sql(config: ConstantVelocityConfig) -> str:
    return (
        f"velocity_history_points >= {config.minimum_history_points} "
        f"AND velocity_history_span_s >= {config.minimum_history_span_seconds}"
    )


def _origin_conditions(
    config: ConstantVelocityConfig,
    horizon_s: int,
) -> list[str]:
    split = config.split.replace("'", "''")
    conditions = [
        f"o.split = '{split}'",
        (
            "CAST(round(o.timeAtServer - b.segment_start_time) AS BIGINT) "
            f"% {config.origin_stride_seconds} = 0"
        ),
        f"o.timeAtServer >= b.segment_start_time + {config.history_seconds}",
        f"o.timeAtServer + {int(horizon_s)} <= b.segment_end_time",
    ]
    if config.exclude_interpolated_origins:
        conditions.append("NOT o.is_interpolated")
    if config.exclude_suspicious_origins:
        conditions.append("NOT o.is_speed_suspicious")
    return conditions
