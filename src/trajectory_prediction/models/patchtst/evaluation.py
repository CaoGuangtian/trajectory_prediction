"""Held-out test evaluation for trained trajectory PatchTST checkpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa

from trajectory_prediction.models.imm_kalman.evaluation import (
    METRIC_COLUMNS,
    SEGMENT_SCHEMA,
    _MetricAccumulator,
    _write_csv_atomic,
    _write_json_atomic,
    _write_parquet_atomic,
)

from .config import PatchTSTConfig
from .data import (
    EARTH_MEAN_RADIUS_M,
    PatchTSTSource,
    TrajectoryWindowCorpus,
    load_normalization_scales,
)
from .runtime import configure_runtime, require_torch, resolve_runtime_device
from .training import _source_entry, _validate_checkpoint


def evaluate_patchtst(
    sources: Iterable[PatchTSTSource],
    config: PatchTSTConfig,
    checkpoint_path: Path,
    normalization_path: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    torch = require_torch()
    from .model import TrajectoryPatchTST, model_parameter_count

    selected_sources = tuple(sorted(sources, key=lambda source: source.subset_id))
    runtime = resolve_runtime_device(
        config.device,
        config.cuda_device_index,
        config.use_mixed_precision,
        torch,
    )
    configure_runtime(config.random_seed, config.cpu_threads, runtime, torch)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PatchTST checkpoint does not exist: {checkpoint_path}")
    scales = load_normalization_scales(normalization_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=runtime.device, weights_only=False
    )
    _validate_checkpoint(checkpoint, config)
    checkpoint_scales = checkpoint.get("normalization")
    if not isinstance(checkpoint_scales, dict) or not np.allclose(
        scales.array,
        np.asarray(
            [
                checkpoint_scales.get("east_m", np.nan),
                checkpoint_scales.get("north_m", np.nan),
                checkpoint_scales.get("up_m", np.nan),
            ],
            dtype=np.float32,
        ),
    ):
        raise ValueError(
            "Normalization artifact does not match the selected checkpoint."
        )
    model = TrajectoryPatchTST(config).to(runtime.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    corpus = TrajectoryWindowCorpus(
        selected_sources, "test", config, config.test_origin_stride_seconds
    )
    if not len(corpus):
        raise RuntimeError("PatchTST test evaluation has no eligible complete windows.")

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.yaml"
    horizon_path = output_directory / "metrics_by_horizon.csv"
    segment_path = output_directory / "metrics_by_segment.parquet"
    existing = [
        path for path in (manifest_path, horizon_path, segment_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "PatchTST evaluation already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    horizons = config.prediction_horizons_seconds
    global_metrics = {horizon: _MetricAccumulator() for horizon in horizons}
    segment_metrics: dict[tuple[int, int], _MetricAccumulator] = {}
    scale_array = scales.array[None, None, :]
    all_indices = np.arange(len(corpus), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(corpus), config.evaluation_batch_size):
            batch = corpus.batch(
                all_indices[start : start + config.evaluation_batch_size], scales
            )
            history = torch.from_numpy(batch.history_enu).to(
                runtime.device, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=runtime.mixed_precision,
            ):
                normalized_prediction = model(history)
            prediction = (
                normalized_prediction.float().cpu().numpy() * scale_array
            ).astype(np.float64)
            _accumulate_batch(
                corpus,
                batch,
                prediction,
                horizons,
                global_metrics,
                segment_metrics,
            )

    horizon_records = [global_metrics[horizon].record(horizon) for horizon in horizons]
    segment_rows = _segment_rows(corpus, horizons, segment_metrics)
    segment_table = pa.Table.from_pylist(segment_rows, schema=SEGMENT_SCHEMA)
    _write_parquet_atomic(segment_path, segment_table, overwrite)
    _write_csv_atomic(
        horizon_path,
        list(METRIC_COLUMNS),
        [tuple(record[column] for column in METRIC_COLUMNS) for record in horizon_records],
        overwrite,
    )
    configuration = config.as_manifest_dict()
    configuration["split"] = "test"
    candidate_predictions = sum(
        int(record["candidate_origins"]) for record in horizon_records
    )
    evaluated_targets = sum(
        int(record["evaluated_targets"]) for record in horizon_records
    )
    manifest: dict[str, object] = {
        "artifact": "patchtst_evaluation",
        "model_type": config.model_type,
        "display_name": config.display_name,
        "evaluation_protocol": "complete-window causal open-loop",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "epoch": int(checkpoint["epoch"]),
            "best_validation_huber": float(checkpoint["best_validation_loss"]),
            "config_hash": checkpoint["config_hash"],
        },
        "normalization": {
            "path": normalization_path.as_posix(),
            **scales.as_dict(),
        },
        "runtime": {
            "requested_device": runtime.requested,
            "resolved_device": runtime.resolved,
            "mixed_precision": runtime.mixed_precision,
            "torch_version": torch.__version__,
        },
        "sources": [_source_entry(source) for source in selected_sources],
        "outputs": {
            "metrics_by_horizon": horizon_path.as_posix(),
            "metrics_by_segment": segment_path.as_posix(),
        },
        "statistics": {
            "selected_subsets": [source.subset_id for source in selected_sources],
            "evaluated_segments": len(
                {segment_index for segment_index, _ in segment_metrics}
            ),
            "segment_horizon_rows": segment_table.num_rows,
            "unique_prediction_origins": len(corpus),
            "candidate_predictions": candidate_predictions,
            "evaluated_targets": evaluated_targets,
            "overall_target_coverage": evaluated_targets / candidate_predictions,
            "model_parameters": model_parameter_count(model),
        },
        "notes": [
            "The model input is the unsmoothed standardized trajectory represented in forecast-origin local ENU coordinates.",
            "Input and output lengths are 60 one-hertz points; every prediction is direct and open-loop.",
            "Normalization was fitted only on train; checkpoint selection used validation; this command alone reads test.",
            "The target remains the canonical standardized future trajectory, not a model-smoothed surrogate.",
            "Interpolated and suspicious target points are excluded by the shared quality mask without dropping the origin.",
            "Horizontal error uses great-circle distance in metres and vertical error uses geometric altitude.",
            "Prediction windows never cross segment_id boundaries.",
            "No per-origin prediction rows or model-input windows are persisted.",
            "JSON syntax is used because JSON is a valid YAML 1.2 subset.",
        ],
    }
    _write_json_atomic(manifest_path, manifest, overwrite)
    return manifest


def _accumulate_batch(
    corpus: TrajectoryWindowCorpus,
    batch: object,
    prediction: np.ndarray,
    horizons: tuple[int, ...],
    global_metrics: dict[int, _MetricAccumulator],
    segment_metrics: dict[tuple[int, int], _MetricAccumulator],
) -> None:
    cosine_latitude = np.maximum(
        np.cos(np.radians(batch.origin_latitude)), 1.0e-6
    )
    for horizon in horizons:
        horizon_index = horizon - 1
        predicted_east = prediction[:, horizon_index, 0]
        predicted_north = prediction[:, horizon_index, 1]
        predicted_up = prediction[:, horizon_index, 2]
        predicted_latitude = batch.origin_latitude + np.degrees(
            predicted_north / EARTH_MEAN_RADIUS_M
        )
        predicted_longitude = batch.origin_longitude + np.degrees(
            predicted_east / (EARTH_MEAN_RADIUS_M * cosine_latitude)
        )
        predicted_altitude = batch.origin_altitude + predicted_up
        valid = batch.target_mask[:, horizon_index]
        horizontal = _haversine_metres(
            predicted_longitude[valid],
            predicted_latitude[valid],
            batch.target_longitude[valid, horizon_index],
            batch.target_latitude[valid, horizon_index],
        )
        vertical = np.abs(
            predicted_altitude[valid] - batch.target_altitude[valid, horizon_index]
        )
        error_3d = np.sqrt(np.square(horizontal) + np.square(vertical))
        global_metrics[horizon].update(len(valid), horizontal, vertical, error_3d)
        for segment_index in np.unique(batch.segment_indices):
            rows = batch.segment_indices == segment_index
            segment_valid = rows & valid
            segment_horizontal = _haversine_metres(
                predicted_longitude[segment_valid],
                predicted_latitude[segment_valid],
                batch.target_longitude[segment_valid, horizon_index],
                batch.target_latitude[segment_valid, horizon_index],
            )
            segment_vertical = np.abs(
                predicted_altitude[segment_valid]
                - batch.target_altitude[segment_valid, horizon_index]
            )
            accumulator = segment_metrics.setdefault(
                (int(segment_index), horizon), _MetricAccumulator()
            )
            accumulator.update(
                int(rows.sum()),
                segment_horizontal,
                segment_vertical,
                np.sqrt(np.square(segment_horizontal) + np.square(segment_vertical)),
            )


def _segment_rows(
    corpus: TrajectoryWindowCorpus,
    horizons: tuple[int, ...],
    accumulators: dict[tuple[int, int], _MetricAccumulator],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (segment_index, horizon), accumulator in sorted(accumulators.items()):
        segment = corpus.segments[segment_index]
        record = accumulator.record(horizon)
        rows.append(
            {
                "subset_id": segment.subset_id,
                "split": "test",
                "aircraft": segment.aircraft,
                "trajectory_id": segment.trajectory_id,
                "segment_id": segment.segment_id,
                "trusted": segment.trusted,
                "segment_start_time": float(segment.times[0]),
                "segment_end_time": float(segment.times[-1]),
                "segment_duration_s": float(segment.times[-1] - segment.times[0]),
                "segment_points": int(segment.times.size),
                "segment_observed_points": int((~segment.is_interpolated).sum()),
                "segment_interpolated_points": int(segment.is_interpolated.sum()),
                **record,
            }
        )
    return rows


def _haversine_metres(
    longitude_a: np.ndarray,
    latitude_a: np.ndarray,
    longitude_b: np.ndarray,
    latitude_b: np.ndarray,
) -> np.ndarray:
    latitude_delta = np.radians(latitude_b - latitude_a)
    longitude_delta = np.radians(longitude_b - longitude_a)
    haversine = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(np.radians(latitude_a))
        * np.cos(np.radians(latitude_b))
        * np.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * EARTH_MEAN_RADIUS_M * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )
