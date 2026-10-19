"""Position-hold baseline evaluation without materializing prediction windows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow.parquet as pq


EARTH_MEAN_RADIUS_M = 6_371_008.8


@dataclass(frozen=True, slots=True)
class PositionHoldSource:
    subset_id: int
    parquet_path: Path
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PositionHoldConfig:
    version: str
    dataset_root: Path
    subsets: tuple[int, ...]
    split: str
    required_history_points: int
    prediction_horizons_seconds: tuple[int, ...]
    origin_stride_seconds: int
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
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "split": self.split,
            "required_history_points": self.required_history_points,
            "history_points": self.required_history_points,
            "prediction_horizons_seconds": list(
                self.prediction_horizons_seconds
            ),
            "origin_stride_seconds": self.origin_stride_seconds,
            "exclude_interpolated_origins": self.exclude_interpolated_origins,
            "exclude_suspicious_origins": self.exclude_suspicious_origins,
            "exclude_interpolated_targets": self.exclude_interpolated_targets,
            "exclude_suspicious_targets": self.exclude_suspicious_targets,
            "output_root": self.output_root.as_posix(),
            "duckdb_memory_limit": self.duckdb_memory_limit,
            "duckdb_threads": self.duckdb_threads,
            "config_hash": self.config_hash,
        }


def load_position_hold_config(path: Path) -> PositionHoldConfig:
    """Load the JSON-compatible YAML configuration."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Position-hold configuration must be an object.")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config = PositionHoldConfig(
        version=str(_required(payload, "version")),
        dataset_root=Path(_required(payload, "dataset_root")),
        subsets=tuple(int(value) for value in _required(payload, "subsets")),
        split=str(_required(payload, "split")),
        required_history_points=int(_required(payload, "required_history_points")),
        prediction_horizons_seconds=tuple(
            int(value)
            for value in _required(payload, "prediction_horizons_seconds")
        ),
        origin_stride_seconds=int(_required(payload, "origin_stride_seconds")),
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


def evaluate_position_hold(
    sources: Iterable[PositionHoldSource],
    config: PositionHoldConfig,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate position hold and write compact aggregate artifacts."""

    selected_sources = tuple(sorted(sources, key=lambda item: item.subset_id))
    if not selected_sources:
        raise ValueError("At least one standardized subset is required.")
    subset_ids = [source.subset_id for source in selected_sources]
    if len(set(subset_ids)) != len(subset_ids):
        raise ValueError("Position-hold sources contain duplicate subset IDs.")
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
            "Position-hold output already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    temporary_directory = tempfile.TemporaryDirectory(
        prefix=".position_hold_duckdb_",
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
        _create_evaluation_view(connection, config)

        segment_table = connection.execute(_segment_metrics_sql()).to_arrow_table()
        horizon_cursor = connection.execute(_horizon_metrics_sql())
        horizon_columns = [description[0] for description in horizon_cursor.description]
        horizon_rows = horizon_cursor.fetchall()

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
        "artifact": "position_hold_baseline_evaluation",
        "model_type": "position_hold",
        "display_name": "Position Hold",
        "evaluation_protocol": "complete-window causal open-loop",
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
            "candidate_predictions": candidate_predictions,
            "evaluated_targets": evaluated_targets,
            "overall_target_coverage": (
                evaluated_targets / candidate_predictions
                if candidate_predictions
                else 0.0
            ),
        },
        "notes": [
            "The prediction at every horizon is the last observed position at the forecast origin.",
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
        raise ValueError(f"Missing position-hold configuration key: {key}")
    return mapping[key]


def _validate_config(config: PositionHoldConfig) -> None:
    if not config.version.strip():
        raise ValueError("version must not be empty.")
    if not config.subsets or any(value < 1 for value in config.subsets):
        raise ValueError("subsets must contain positive integers.")
    if config.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if config.required_history_points < 1:
        raise ValueError("required_history_points must be positive.")
    horizons = config.prediction_horizons_seconds
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("prediction_horizons_seconds must contain positive integers.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError(
            "prediction_horizons_seconds must be unique and strictly increasing."
        )
    if config.origin_stride_seconds < 1:
        raise ValueError("origin_stride_seconds must be positive.")
    if not config.duckdb_memory_limit.strip():
        raise ValueError("duckdb_memory_limit must not be empty.")
    if config.duckdb_threads < 1:
        raise ValueError("duckdb_threads must be positive.")


def _create_source_view(
    connection: duckdb.DuckDBPyConnection,
    sources: tuple[PositionHoldSource, ...],
) -> None:
    selects: list[str] = []
    for source in sources:
        path = str(source.parquet_path.resolve()).replace("'", "''")
        selects.append(
            f"""
            SELECT
                {int(source.subset_id)}::INTEGER AS subset_id,
                aircraft,
                trajectory_id,
                segment_id,
                timeAtServer,
                longitude,
                latitude,
                geoAltitude,
                baroAltitude,
                trusted,
                split,
                is_interpolated,
                is_speed_suspicious
            FROM read_parquet('{path}')
            """
        )
    connection.execute(
        "CREATE TEMP VIEW source_points AS " + " UNION ALL ".join(selects)
    )


def _create_evaluation_view(
    connection: duckdb.DuckDBPyConnection,
    config: PositionHoldConfig,
) -> None:
    horizons = ", ".join(
        f"({int(value)})" for value in config.prediction_horizons_seconds
    )
    split = config.split.replace("'", "''")
    origin_conditions = [
        f"o.split = '{split}'",
        (
            "CAST(round(o.timeAtServer - b.segment_start_time) AS BIGINT) "
            f"% {config.origin_stride_seconds} = 0"
        ),
        (
            "o.timeAtServer >= b.segment_start_time + "
            f"{config.required_history_points - 1}"
        ),
        (
            "o.timeAtServer + "
            f"{max(config.prediction_horizons_seconds)} <= b.segment_end_time"
        ),
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
                    pow(sin(radians(t.latitude - c.origin_latitude) / 2.0), 2)
                    + cos(radians(c.origin_latitude)) * cos(radians(t.latitude))
                    * pow(sin(radians(t.longitude - c.origin_longitude) / 2.0), 2)
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
                o.longitude AS origin_longitude,
                o.latitude AS origin_latitude,
                o.geoAltitude AS origin_geo_altitude,
                o.baroAltitude AS origin_baro_altitude,
                h.horizon_s
            FROM source_points AS o
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
                THEN abs(t.geoAltitude - c.origin_geo_altitude)
            END AS vertical_error_m,
            CASE WHEN {target_valid} THEN sqrt(
                pow({horizontal_error}, 2)
                + pow(t.geoAltitude - c.origin_geo_altitude, 2)
            ) END AS error_3d_m,
            CASE WHEN {target_valid}
                AND c.origin_baro_altitude IS NOT NULL
                AND t.baroAltitude IS NOT NULL
                THEN abs(t.baroAltitude - c.origin_baro_altitude)
            END AS barometric_vertical_error_m
        FROM candidates AS c
        LEFT JOIN source_points AS t
          ON t.subset_id = c.subset_id
         AND t.segment_id = c.segment_id
         AND t.timeAtServer = c.origin_time + c.horizon_s
        """
    )


def _segment_metrics_sql() -> str:
    return """
        SELECT
            subset_id,
            split,
            aircraft,
            trajectory_id,
            segment_id,
            any_value(trusted) AS trusted,
            any_value(segment_start_time) AS segment_start_time,
            any_value(segment_end_time) AS segment_end_time,
            any_value(segment_duration_s) AS segment_duration_s,
            any_value(segment_points)::BIGINT AS segment_points,
            any_value(segment_observed_points)::BIGINT AS segment_observed_points,
            any_value(segment_interpolated_points)::BIGINT AS segment_interpolated_points,
            horizon_s,
            count(*)::BIGINT AS candidate_origins,
            count(horizontal_error_m)::BIGINT AS evaluated_targets,
            count(horizontal_error_m)::DOUBLE / count(*) AS coverage,
            avg(horizontal_error_m) AS horizontal_mean_m,
            sqrt(avg(pow(horizontal_error_m, 2))) AS horizontal_rmse_m,
            median(horizontal_error_m) AS horizontal_median_m,
            quantile_cont(horizontal_error_m, 0.90) AS horizontal_p90_m,
            quantile_cont(horizontal_error_m, 0.95) AS horizontal_p95_m,
            avg(vertical_error_m) AS vertical_mae_m,
            sqrt(avg(pow(vertical_error_m, 2))) AS vertical_rmse_m,
            avg(error_3d_m) AS error_3d_mean_m,
            sqrt(avg(pow(error_3d_m, 2))) AS error_3d_rmse_m,
            count(barometric_vertical_error_m)::BIGINT AS barometric_evaluated_targets,
            avg(barometric_vertical_error_m) AS barometric_vertical_mae_m,
            sqrt(avg(pow(barometric_vertical_error_m, 2))) AS barometric_vertical_rmse_m
        FROM evaluation_rows
        GROUP BY
            subset_id,
            split,
            aircraft,
            trajectory_id,
            segment_id,
            horizon_s
        ORDER BY subset_id, aircraft, trajectory_id, segment_id, horizon_s
    """


def _horizon_metrics_sql() -> str:
    return """
        SELECT
            horizon_s,
            count(*)::BIGINT AS candidate_origins,
            count(horizontal_error_m)::BIGINT AS evaluated_targets,
            count(horizontal_error_m)::DOUBLE / count(*) AS coverage,
            avg(horizontal_error_m) AS horizontal_mean_m,
            sqrt(avg(pow(horizontal_error_m, 2))) AS horizontal_rmse_m,
            median(horizontal_error_m) AS horizontal_median_m,
            quantile_cont(horizontal_error_m, 0.90) AS horizontal_p90_m,
            quantile_cont(horizontal_error_m, 0.95) AS horizontal_p95_m,
            avg(vertical_error_m) AS vertical_mae_m,
            sqrt(avg(pow(vertical_error_m, 2))) AS vertical_rmse_m,
            avg(error_3d_m) AS error_3d_mean_m,
            sqrt(avg(pow(error_3d_m, 2))) AS error_3d_rmse_m,
            count(barometric_vertical_error_m)::BIGINT AS barometric_evaluated_targets,
            avg(barometric_vertical_error_m) AS barometric_vertical_mae_m,
            sqrt(avg(pow(barometric_vertical_error_m, 2))) AS barometric_vertical_rmse_m
        FROM evaluation_rows
        GROUP BY horizon_s
        ORDER BY horizon_s
    """


def _source_manifest_entry(source: PositionHoldSource) -> dict[str, object]:
    entry: dict[str, object] = {
        "subset": source.subset_id,
        "path": source.parquet_path.as_posix(),
    }
    if source.manifest_path is not None and source.manifest_path.is_file():
        payload = json.loads(source.manifest_path.read_text(encoding="utf-8"))
        entry["manifest"] = source.manifest_path.as_posix()
        entry["preprocessing_version"] = payload.get("configuration", {}).get(
            "version"
        )
        entry["preprocessing_config_hash"] = payload.get("configuration", {}).get(
            "config_hash"
        )
    return entry


def _write_parquet_atomic(path: Path, table: object, overwrite: bool) -> None:
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(
    path: Path,
    columns: list[str],
    rows: list[tuple[object, ...]],
    overwrite: bool,
) -> None:
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        writer = csv.writer(temporary)
        writer.writerow(columns)
        writer.writerows(rows)
        temporary.close()
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(
    path: Path,
    payload: dict[str, object],
    overwrite: bool,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
