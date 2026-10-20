"""Held-out evaluation for IMM-PatchTST residual fusion."""

from __future__ import annotations

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
from trajectory_prediction.models.patchtst.data import (
    PatchTSTSource,
    load_normalization_scales,
)
from trajectory_prediction.models.patchtst.evaluation import (
    _accumulate_batch,
    _segment_rows,
)
from trajectory_prediction.models.patchtst.runtime import (
    configure_runtime,
    require_torch,
    resolve_runtime_device,
)

from .cache import build_corpus_for_split, open_imm_prediction_cache
from .config import ImmPatchTSTConfig
from .training import _source_entry, _validate_checkpoint


def evaluate_imm_patchtst(
    sources: Iterable[PatchTSTSource],
    config: ImmPatchTSTConfig,
    test_cache_directory: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    torch = require_torch()
    from .model import ImmPatchTSTResidual, model_parameter_count

    selected_sources = tuple(sorted(sources, key=lambda source: source.subset_id))
    patch_config = config.patchtst
    runtime = resolve_runtime_device(
        patch_config.device,
        patch_config.cuda_device_index,
        patch_config.use_mixed_precision,
        torch,
    )
    configure_runtime(
        patch_config.random_seed, patch_config.cpu_threads, runtime, torch
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"IMM-PatchTST checkpoint does not exist: {checkpoint_path}"
        )
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
    model = ImmPatchTSTResidual(config).to(runtime.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    corpus = build_corpus_for_split(selected_sources, "test", config)
    if not len(corpus):
        raise RuntimeError("IMM-PatchTST test evaluation has no eligible windows.")
    cache = open_imm_prediction_cache(
        test_cache_directory, corpus, "test", config
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
            "IMM-PatchTST evaluation already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    horizons = patch_config.prediction_horizons_seconds
    global_metrics = {horizon: _MetricAccumulator() for horizon in horizons}
    segment_metrics: dict[tuple[int, int], _MetricAccumulator] = {}
    scale_array = scales.array[None, None, :]
    all_indices = np.arange(len(corpus), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(corpus), patch_config.evaluation_batch_size):
            indices = all_indices[
                start : start + patch_config.evaluation_batch_size
            ]
            batch = corpus.batch(indices, scales)
            history = torch.from_numpy(batch.history_enu).to(
                runtime.device, non_blocking=True
            )
            imm_prediction = np.asarray(
                cache.positions_enu[indices], dtype=np.float32
            ) / scale_array
            imm_tensor = torch.from_numpy(imm_prediction).to(
                runtime.device, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=runtime.mixed_precision,
            ):
                normalized_prediction = model(history, imm_tensor)
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
        "artifact": "imm_patchtst_evaluation",
        "model_type": config.model_type,
        "display_name": config.display_name,
        "evaluation_protocol": "complete-window causal open-loop residual fusion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "patchtst_configuration": patch_config.as_manifest_dict(),
        "imm_configuration": config.imm.as_manifest_dict(),
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
        "imm_cache": {
            "directory": cache.directory.as_posix(),
            "window_count": cache.metadata["window_count"],
            "corpus_fingerprint": cache.metadata["corpus_fingerprint"],
            "imm_config_hash": cache.metadata["imm_config_hash"],
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
            "The frozen Standard IMM consumes raw metre-scale history and causal quality masks.",
            "PatchTST predicts a normalized residual which is added to the cached IMM forecast.",
            "The IMM and PatchTST branches use exactly the same 60-point history and forecast origin.",
            "Checkpoint selection used validation; this command alone reads test.",
            "Interpolated and suspicious targets are excluded by the shared quality mask.",
            "No per-origin fused prediction rows or model-input windows are persisted.",
        ],
    }
    _write_json_atomic(manifest_path, manifest, overwrite)
    return manifest
