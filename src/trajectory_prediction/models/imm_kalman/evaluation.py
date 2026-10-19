"""Resource-bounded evaluation of causal IMM trajectory predictions."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import ImmKalmanConfig
from .core import MODEL_NAMES, predict_imm_batch


EARTH_MEAN_RADIUS_M = 6_371_008.8


SOURCE_COLUMNS = (
    "aircraft",
    "trajectory_id",
    "segment_id",
    "point_index",
    "timeAtServer",
    "longitude",
    "latitude",
    "geoAltitude",
    "baroAltitude",
    "trusted",
    "split",
    "is_interpolated",
    "is_speed_suspicious",
)

METRIC_COLUMNS = (
    "horizon_s",
    "candidate_origins",
    "evaluated_targets",
    "coverage",
    "horizontal_mean_m",
    "horizontal_rmse_m",
    "horizontal_median_m",
    "horizontal_p90_m",
    "horizontal_p95_m",
    "vertical_mae_m",
    "vertical_rmse_m",
    "error_3d_mean_m",
    "error_3d_rmse_m",
    "barometric_evaluated_targets",
    "barometric_vertical_mae_m",
    "barometric_vertical_rmse_m",
)

SEGMENT_SCHEMA = pa.schema(
    [
        pa.field("subset_id", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("aircraft", pa.int64(), nullable=False),
        pa.field("trajectory_id", pa.int64(), nullable=False),
        pa.field("segment_id", pa.int64(), nullable=False),
        pa.field("trusted", pa.string(), nullable=False),
        pa.field("segment_start_time", pa.float64(), nullable=False),
        pa.field("segment_end_time", pa.float64(), nullable=False),
        pa.field("segment_duration_s", pa.float64(), nullable=False),
        pa.field("segment_points", pa.int64(), nullable=False),
        pa.field("segment_observed_points", pa.int64(), nullable=False),
        pa.field("segment_interpolated_points", pa.int64(), nullable=False),
        pa.field("horizon_s", pa.int32(), nullable=False),
        pa.field("candidate_origins", pa.int64(), nullable=False),
        pa.field("evaluated_targets", pa.int64(), nullable=False),
        pa.field("coverage", pa.float64(), nullable=False),
        pa.field("horizontal_mean_m", pa.float64()),
        pa.field("horizontal_rmse_m", pa.float64()),
        pa.field("horizontal_median_m", pa.float64()),
        pa.field("horizontal_p90_m", pa.float64()),
        pa.field("horizontal_p95_m", pa.float64()),
        pa.field("vertical_mae_m", pa.float64()),
        pa.field("vertical_rmse_m", pa.float64()),
        pa.field("error_3d_mean_m", pa.float64()),
        pa.field("error_3d_rmse_m", pa.float64()),
        pa.field("barometric_evaluated_targets", pa.int64(), nullable=False),
        pa.field("barometric_vertical_mae_m", pa.float64()),
        pa.field("barometric_vertical_rmse_m", pa.float64()),
    ]
)


@dataclass(frozen=True, slots=True)
class ImmKalmanSource:
    subset_id: int
    parquet_path: Path
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _Segment:
    subset_id: int
    aircraft: int
    trajectory_id: int
    segment_id: int
    trusted: str
    split: str
    times: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    geo_altitude: np.ndarray
    baro_altitude: np.ndarray
    is_interpolated: np.ndarray
    is_suspicious: np.ndarray


@dataclass(slots=True)
class _MetricAccumulator:
    candidate_origins: int = 0
    horizontal_parts: list[np.ndarray] = field(default_factory=list)
    vertical_parts: list[np.ndarray] = field(default_factory=list)
    error_3d_parts: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        candidate_count: int,
        horizontal_error: np.ndarray,
        vertical_error: np.ndarray,
        error_3d: np.ndarray,
    ) -> None:
        self.candidate_origins += int(candidate_count)
        if horizontal_error.size:
            self.horizontal_parts.append(np.asarray(horizontal_error, dtype=float))
            self.vertical_parts.append(np.asarray(vertical_error, dtype=float))
            self.error_3d_parts.append(np.asarray(error_3d, dtype=float))

    def record(self, horizon_s: int) -> dict[str, object]:
        horizontal = _concatenate(self.horizontal_parts)
        vertical = _concatenate(self.vertical_parts)
        error_3d = _concatenate(self.error_3d_parts)
        evaluated = int(horizontal.size)
        return {
            "horizon_s": int(horizon_s),
            "candidate_origins": int(self.candidate_origins),
            "evaluated_targets": evaluated,
            "coverage": (
                evaluated / self.candidate_origins
                if self.candidate_origins
                else 0.0
            ),
            "horizontal_mean_m": _mean(horizontal),
            "horizontal_rmse_m": _rmse(horizontal),
            "horizontal_median_m": _quantile(horizontal, 0.50),
            "horizontal_p90_m": _quantile(horizontal, 0.90),
            "horizontal_p95_m": _quantile(horizontal, 0.95),
            "vertical_mae_m": _mean(vertical),
            "vertical_rmse_m": _rmse(vertical),
            "error_3d_mean_m": _mean(error_3d),
            "error_3d_rmse_m": _rmse(error_3d),
            "barometric_evaluated_targets": 0,
            "barometric_vertical_mae_m": None,
            "barometric_vertical_rmse_m": None,
        }


def evaluate_imm_kalman(
    sources: Iterable[ImmKalmanSource],
    config: ImmKalmanConfig,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate IMM-CV-CT-Singer without persisting per-origin predictions."""

    selected_sources = tuple(sorted(sources, key=lambda item: item.subset_id))
    if not selected_sources:
        raise ValueError("At least one standardized subset is required.")
    subset_ids = [source.subset_id for source in selected_sources]
    if len(set(subset_ids)) != len(subset_ids):
        raise ValueError("IMM-Kalman sources contain duplicate subset IDs.")
    for source in selected_sources:
        if not source.parquet_path.is_file():
            raise FileNotFoundError(
                f"Standardized trajectory artifact does not exist: {source.parquet_path}"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.yaml"
    horizon_path = output_directory / "metrics_by_horizon.csv"
    segment_path = output_directory / "metrics_by_segment.parquet"
    existing = [
        path for path in (manifest_path, horizon_path, segment_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "IMM-Kalman output already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    horizons = config.prediction_horizons_seconds
    global_metrics = {value: _MetricAccumulator() for value in horizons}
    segment_rows: list[dict[str, object]] = []
    evaluated_segment_keys: set[tuple[int, int]] = set()
    unique_origins = 0
    fallback_origins = 0
    probability_sum = np.zeros(len(MODEL_NAMES), dtype=float)
    probability_count = 0
    maneuver_origins = 0
    turn_maneuver_origins = 0
    acceleration_maneuver_origins = 0

    for source in selected_sources:
        for segment in _iter_segments(source, config):
            result = _evaluate_segment(segment, config)
            if result is None:
                continue
            (
                rows,
                accumulators,
                origin_count,
                fallback_count,
                model_probability_sum,
                model_probability_count,
                gate_counts,
            ) = result
            segment_rows.extend(rows)
            unique_origins += origin_count
            fallback_origins += fallback_count
            probability_sum += model_probability_sum
            probability_count += model_probability_count
            maneuver_origins += int(gate_counts[0])
            turn_maneuver_origins += int(gate_counts[1])
            acceleration_maneuver_origins += int(gate_counts[2])
            evaluated_segment_keys.add((segment.subset_id, segment.segment_id))
            for horizon_s, accumulator in accumulators.items():
                record = accumulator.record(horizon_s)
                global_metrics[horizon_s].update(
                    int(record["candidate_origins"]),
                    _concatenate(accumulator.horizontal_parts),
                    _concatenate(accumulator.vertical_parts),
                    _concatenate(accumulator.error_3d_parts),
                )

    if not segment_rows:
        raise RuntimeError(
            "IMM-Kalman evaluation produced no eligible complete prediction windows."
        )

    horizon_records = [global_metrics[value].record(value) for value in horizons]
    segment_table = pa.Table.from_pylist(segment_rows, schema=SEGMENT_SCHEMA)
    _write_parquet_atomic(segment_path, segment_table, overwrite)
    _write_csv_atomic(
        horizon_path,
        list(METRIC_COLUMNS),
        [[record[column] for column in METRIC_COLUMNS] for record in horizon_records],
        overwrite,
    )

    candidate_predictions = sum(
        int(record["candidate_origins"]) for record in horizon_records
    )
    evaluated_targets = sum(
        int(record["evaluated_targets"]) for record in horizon_records
    )
    mean_probabilities = (
        probability_sum / probability_count
        if probability_count
        else np.zeros(len(MODEL_NAMES), dtype=float)
    )
    manifest: dict[str, object] = {
        "artifact": "imm_kalman_model_evaluation",
        "model_type": "imm_kalman",
        "display_name": (
            "IMM · gated CV/CT/Singer"
            if config.motion_gate_enabled
            else "IMM · CV/CT/Singer"
        ),
        "evaluation_protocol": "complete-window causal open-loop",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.as_manifest_dict(),
        "sources": [_source_manifest_entry(source) for source in selected_sources],
        "outputs": {
            "metrics_by_horizon": horizon_path.as_posix(),
            "metrics_by_segment": segment_path.as_posix(),
        },
        "statistics": {
            "selected_subsets": subset_ids,
            "evaluated_segments": len(evaluated_segment_keys),
            "segment_horizon_rows": segment_table.num_rows,
            "unique_prediction_origins": unique_origins,
            "constant_velocity_fallback_origins": fallback_origins,
            "constant_velocity_fallback_rate": (
                fallback_origins / unique_origins if unique_origins else 0.0
            ),
            "maneuver_gated_origins": maneuver_origins,
            "maneuver_gated_rate": (
                maneuver_origins / unique_origins if unique_origins else 0.0
            ),
            "turn_maneuver_origins": turn_maneuver_origins,
            "turn_maneuver_rate": (
                turn_maneuver_origins / unique_origins if unique_origins else 0.0
            ),
            "acceleration_maneuver_origins": acceleration_maneuver_origins,
            "acceleration_maneuver_rate": (
                acceleration_maneuver_origins / unique_origins
                if unique_origins
                else 0.0
            ),
            "ols_cv_routed_origins": (
                unique_origins - maneuver_origins
                if config.motion_gate_enabled
                else 0
            ),
            "candidate_predictions": candidate_predictions,
            "evaluated_targets": evaluated_targets,
            "overall_target_coverage": (
                evaluated_targets / candidate_predictions
                if candidate_predictions
                else 0.0
            ),
            "mean_model_probability_cv": float(mean_probabilities[0]),
            "mean_model_probability_ct": float(mean_probabilities[1]),
            "mean_model_probability_singer": float(mean_probabilities[2]),
        },
        "notes": [
            "Every forecast uses exactly history_seconds one-hertz points ending at the origin.",
            "Every candidate origin has a complete window through the largest configured horizon.",
            "IMM parameters and posterior state use only observations at or before the forecast origin.",
            "Forecasting is strictly open-loop; no future truth updates the filter.",
            "When enabled, the causal maneuver gate routes non-maneuver origins to a recent-window least-squares constant-velocity forecast.",
            "CT and Singer are allowed only after two consecutive smoothed history blocks show directionally consistent turn or acceleration evidence.",
            "Interpolated or suspicious history points can be skipped as measurement updates.",
            "Insufficient history falls back to causal constant velocity without dropping the origin.",
            "Horizontal error uses the WGS84 great-circle distance in metres.",
            "Barometric-altitude metrics are retained as null-compatible columns; this model predicts geometric altitude only.",
            "Prediction windows never cross segment_id boundaries.",
            "No per-origin prediction rows or model-input windows are persisted.",
            "JSON syntax is used because JSON is a valid YAML 1.2 subset.",
        ],
    }
    _write_json_atomic(manifest_path, manifest, overwrite)
    return manifest


def _evaluate_segment(
    segment: _Segment,
    config: ImmKalmanConfig,
) -> tuple[
    list[dict[str, object]],
    dict[int, _MetricAccumulator],
    int,
    int,
    np.ndarray,
    int,
    np.ndarray,
] | None:
    history_points = config.history_seconds
    maximum_horizon = max(config.prediction_horizons_seconds)
    point_count = segment.times.size
    first_origin = history_points - 1
    last_origin = point_count - maximum_horizon - 1
    if last_origin < first_origin:
        return None

    origins = np.arange(first_origin, last_origin + 1, dtype=int)
    stride_offsets = np.rint(
        segment.times[origins] - segment.times[0]
    ).astype(np.int64)
    origin_mask = stride_offsets % config.origin_stride_seconds == 0
    if config.exclude_interpolated_origins:
        origin_mask &= ~segment.is_interpolated[origins]
    if config.exclude_suspicious_origins:
        origin_mask &= ~segment.is_suspicious[origins]
    origins = origins[origin_mask]
    if origins.size == 0:
        return None

    accumulators = {
        value: _MetricAccumulator()
        for value in config.prediction_horizons_seconds
    }
    fallback_count = 0
    probability_sum = np.zeros(len(MODEL_NAMES), dtype=float)
    probability_count = 0
    gate_counts = np.zeros(3, dtype=np.int64)
    horizon_array = np.asarray(config.prediction_horizons_seconds, dtype=int)

    for start in range(0, origins.size, config.prediction_batch_size):
        batch_origins = origins[start : start + config.prediction_batch_size]
        window_indices = (
            batch_origins[:, None]
            - (history_points - 1)
            + np.arange(history_points, dtype=int)[None, :]
        )
        origin_longitude = segment.longitude[batch_origins]
        origin_latitude = segment.latitude[batch_origins]
        origin_altitude = segment.geo_altitude[batch_origins]
        cosine_latitude = np.maximum(
            np.cos(np.radians(origin_latitude)),
            1.0e-6,
        )
        history_east = (
            np.radians(
                segment.longitude[window_indices] - origin_longitude[:, None]
            )
            * EARTH_MEAN_RADIUS_M
            * cosine_latitude[:, None]
        )
        history_north = (
            np.radians(
                segment.latitude[window_indices] - origin_latitude[:, None]
            )
            * EARTH_MEAN_RADIUS_M
        )
        history_up = (
            segment.geo_altitude[window_indices] - origin_altitude[:, None]
        )
        histories = np.stack([history_east, history_north, history_up], axis=2)
        update_mask = np.ones(window_indices.shape, dtype=bool)
        if config.skip_interpolated_history_updates:
            update_mask &= ~segment.is_interpolated[window_indices]
        if config.skip_suspicious_history_updates:
            update_mask &= ~segment.is_suspicious[window_indices]

        prediction = predict_imm_batch(
            histories,
            update_mask,
            config.prediction_horizons_seconds,
            config,
        )
        fallback_count += int(prediction.fallback_mask.sum())
        gate_counts += np.array(
            [
                prediction.maneuver_mask.sum(),
                prediction.turn_maneuver_mask.sum(),
                prediction.acceleration_maneuver_mask.sum(),
            ],
            dtype=np.int64,
        )
        usable_probabilities = prediction.issue_model_probabilities[
            ~prediction.fallback_mask
        ]
        if usable_probabilities.size:
            probability_sum += usable_probabilities.sum(axis=0)
            probability_count += usable_probabilities.shape[0]

        predicted_east = prediction.positions_enu[:, :, 0]
        predicted_north = prediction.positions_enu[:, :, 1]
        predicted_up = prediction.positions_enu[:, :, 2]
        predicted_latitude = origin_latitude[:, None] + np.degrees(
            predicted_north / EARTH_MEAN_RADIUS_M
        )
        predicted_longitude = origin_longitude[:, None] + np.degrees(
            predicted_east
            / (EARTH_MEAN_RADIUS_M * cosine_latitude[:, None])
        )
        predicted_altitude = origin_altitude[:, None] + predicted_up

        target_indices = batch_origins[:, None] + horizon_array[None, :]
        for horizon_index, horizon_s in enumerate(config.prediction_horizons_seconds):
            target_index = target_indices[:, horizon_index]
            valid_target = np.ones(batch_origins.size, dtype=bool)
            if config.exclude_interpolated_targets:
                valid_target &= ~segment.is_interpolated[target_index]
            if config.exclude_suspicious_targets:
                valid_target &= ~segment.is_suspicious[target_index]
            horizontal = _haversine_m(
                predicted_longitude[:, horizon_index],
                predicted_latitude[:, horizon_index],
                segment.longitude[target_index],
                segment.latitude[target_index],
            )
            vertical = np.abs(
                predicted_altitude[:, horizon_index]
                - segment.geo_altitude[target_index]
            )
            error_3d = np.sqrt(np.square(horizontal) + np.square(vertical))
            finite = (
                np.isfinite(horizontal)
                & np.isfinite(vertical)
                & np.isfinite(error_3d)
            )
            valid_target &= finite
            accumulators[horizon_s].update(
                batch_origins.size,
                horizontal[valid_target],
                vertical[valid_target],
                error_3d[valid_target],
            )

    metadata = {
        "subset_id": segment.subset_id,
        "split": segment.split,
        "aircraft": segment.aircraft,
        "trajectory_id": segment.trajectory_id,
        "segment_id": segment.segment_id,
        "trusted": segment.trusted,
        "segment_start_time": float(segment.times[0]),
        "segment_end_time": float(segment.times[-1]),
        "segment_duration_s": float(segment.times[-1] - segment.times[0]),
        "segment_points": int(point_count),
        "segment_observed_points": int((~segment.is_interpolated).sum()),
        "segment_interpolated_points": int(segment.is_interpolated.sum()),
    }
    rows = []
    for horizon_s in config.prediction_horizons_seconds:
        row = dict(metadata)
        row.update(accumulators[horizon_s].record(horizon_s))
        rows.append(row)
    return (
        rows,
        accumulators,
        int(origins.size),
        fallback_count,
        probability_sum,
        probability_count,
        gate_counts,
    )


def _iter_segments(
    source: ImmKalmanSource,
    config: ImmKalmanConfig,
) -> Iterator[_Segment]:
    parquet = pq.ParquetFile(source.parquet_path)
    current_key: tuple[int, int, int] | None = None
    pieces: list[pa.RecordBatch] = []
    trusted_set = (
        pa.array(config.trusted_values, type=pa.string())
        if config.trusted_values
        else None
    )

    for batch in parquet.iter_batches(
        batch_size=config.parquet_batch_rows,
        columns=list(SOURCE_COLUMNS),
    ):
        mask = pc.equal(batch.column(batch.schema.get_field_index("split")), config.split)
        if trusted_set is not None:
            trusted_mask = pc.is_in(
                batch.column(batch.schema.get_field_index("trusted")),
                value_set=trusted_set,
            )
            mask = pc.and_(mask, trusted_mask)
        filtered = batch.filter(mask)
        if filtered.num_rows == 0:
            continue
        aircraft = filtered.column(
            filtered.schema.get_field_index("aircraft")
        ).to_numpy(zero_copy_only=False)
        trajectory = filtered.column(
            filtered.schema.get_field_index("trajectory_id")
        ).to_numpy(zero_copy_only=False)
        segment = filtered.column(
            filtered.schema.get_field_index("segment_id")
        ).to_numpy(zero_copy_only=False)
        changes = np.flatnonzero(
            (aircraft[1:] != aircraft[:-1])
            | (trajectory[1:] != trajectory[:-1])
            | (segment[1:] != segment[:-1])
        ) + 1
        boundaries = np.concatenate(
            [np.array([0]), changes, np.array([filtered.num_rows])]
        )
        for index in range(boundaries.size - 1):
            start = int(boundaries[index])
            stop = int(boundaries[index + 1])
            key = (
                int(aircraft[start]),
                int(trajectory[start]),
                int(segment[start]),
            )
            if current_key is not None and key != current_key:
                yield _segment_from_batches(source.subset_id, pieces)
                pieces = []
            current_key = key
            pieces.append(filtered.slice(start, stop - start))
    if pieces:
        yield _segment_from_batches(source.subset_id, pieces)


def _segment_from_batches(
    subset_id: int,
    batches: list[pa.RecordBatch],
) -> _Segment:
    table = pa.Table.from_batches(batches).combine_chunks()

    def numpy(name: str, dtype: object | None = None) -> np.ndarray:
        values = table[name].to_numpy(zero_copy_only=False)
        return values.astype(dtype, copy=False) if dtype is not None else values

    barometric = table["baroAltitude"].to_numpy(zero_copy_only=False)
    return _Segment(
        subset_id=subset_id,
        aircraft=int(table["aircraft"][0].as_py()),
        trajectory_id=int(table["trajectory_id"][0].as_py()),
        segment_id=int(table["segment_id"][0].as_py()),
        trusted=str(table["trusted"][0].as_py()),
        split=str(table["split"][0].as_py()),
        times=numpy("timeAtServer", float),
        longitude=numpy("longitude", float),
        latitude=numpy("latitude", float),
        geo_altitude=numpy("geoAltitude", float),
        baro_altitude=np.asarray(barometric, dtype=float),
        is_interpolated=numpy("is_interpolated", bool),
        is_suspicious=numpy("is_speed_suspicious", bool),
    )


def _haversine_m(
    predicted_longitude: np.ndarray,
    predicted_latitude: np.ndarray,
    target_longitude: np.ndarray,
    target_latitude: np.ndarray,
) -> np.ndarray:
    latitude_delta = np.radians(target_latitude - predicted_latitude)
    longitude_delta = np.radians(target_longitude - predicted_longitude)
    predicted_latitude_radians = np.radians(predicted_latitude)
    target_latitude_radians = np.radians(target_latitude)
    value = (
        np.square(np.sin(latitude_delta / 2.0))
        + np.cos(predicted_latitude_radians)
        * np.cos(target_latitude_radians)
        * np.square(np.sin(longitude_delta / 2.0))
    )
    return 2.0 * EARTH_MEAN_RADIUS_M * np.arcsin(
        np.sqrt(np.minimum(1.0, value))
    )


def _concatenate(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=float)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _rmse(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values.size else None


def _source_manifest_entry(source: ImmKalmanSource) -> dict[str, object]:
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


def _write_parquet_atomic(
    path: Path,
    table: pa.Table,
    overwrite: bool,
) -> None:
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
    rows: list[list[object]],
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
