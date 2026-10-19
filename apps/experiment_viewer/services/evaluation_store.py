"""Read-only discovery and access for model-evaluation artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb


@dataclass(frozen=True)
class EvaluationMetricDefinition:
    key: str
    label: str
    axis_title: str


EVALUATION_METRICS = (
    EvaluationMetricDefinition(
        "horizontal_mean_m", "Mean horizontal error", "Horizontal error (m)"
    ),
    EvaluationMetricDefinition(
        "horizontal_rmse_m", "Horizontal RMSE", "Horizontal RMSE (m)"
    ),
    EvaluationMetricDefinition(
        "horizontal_median_m", "Median horizontal error", "Horizontal error (m)"
    ),
    EvaluationMetricDefinition(
        "horizontal_p90_m", "Horizontal P90", "Horizontal error (m)"
    ),
    EvaluationMetricDefinition(
        "horizontal_p95_m", "Horizontal P95", "Horizontal error (m)"
    ),
    EvaluationMetricDefinition(
        "vertical_mae_m", "Geometric-altitude MAE", "Altitude error (m)"
    ),
    EvaluationMetricDefinition(
        "vertical_rmse_m", "Geometric-altitude RMSE", "Altitude RMSE (m)"
    ),
    EvaluationMetricDefinition(
        "barometric_vertical_mae_m",
        "Barometric-altitude MAE",
        "Altitude error (m)",
    ),
    EvaluationMetricDefinition(
        "barometric_vertical_rmse_m",
        "Barometric-altitude RMSE",
        "Altitude RMSE (m)",
    ),
    EvaluationMetricDefinition(
        "error_3d_mean_m", "Mean 3D error", "3D error (m)"
    ),
    EvaluationMetricDefinition(
        "error_3d_rmse_m", "3D RMSE", "3D RMSE (m)"
    ),
)
EVALUATION_METRIC_BY_KEY = {metric.key: metric for metric in EVALUATION_METRICS}


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    label: str
    model_type: str
    version: str
    split: str
    subset_ids: tuple[int, ...]
    horizons: tuple[int, ...]
    manifest_path: Path
    horizon_metrics_path: Path
    segment_metrics_path: Path
    evaluated_segments: int
    evaluation_protocol: str


class EvaluationStore:
    """Discover compatible experiment outputs without loading them in full."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()

    @lru_cache(maxsize=1)
    def runs(self) -> tuple[EvaluationRun, ...]:
        if not self.output_root.is_dir():
            return ()
        discovered: list[EvaluationRun] = []
        for manifest_path in sorted(self.output_root.rglob("manifest.yaml")):
            run = self._read_run(manifest_path)
            if run is not None:
                discovered.append(run)
        return tuple(discovered)

    def run(self, run_id: str) -> EvaluationRun:
        for run in self.runs():
            if run.run_id == run_id:
                return run
        raise KeyError(f"Unknown evaluation run: {run_id}")

    @lru_cache(maxsize=32)
    def horizon_metrics(self, run_id: str) -> tuple[dict[str, Any], ...]:
        run = self.run(run_id)
        path = run.horizon_metrics_path
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return tuple(_convert_metric_row(row) for row in csv.DictReader(handle))
        with duckdb.connect(database=":memory:") as connection:
            cursor = connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY horizon_s", [str(path)]
            )
            columns = [description[0] for description in cursor.description]
            return tuple(
                dict(zip(columns, row, strict=True)) for row in cursor.fetchall()
            )

    @lru_cache(maxsize=256)
    def segment_metric_values(
        self,
        run_id: str,
        metric_key: str,
        horizon_s: int,
    ) -> tuple[float, ...]:
        if metric_key not in EVALUATION_METRIC_BY_KEY:
            raise ValueError(f"Unsupported evaluation metric: {metric_key}")
        run = self.run(run_id)
        with duckdb.connect(database=":memory:") as connection:
            rows = connection.execute(
                f"""
                SELECT {metric_key}
                FROM read_parquet(?)
                WHERE horizon_s = ? AND {metric_key} IS NOT NULL
                ORDER BY {metric_key}
                """,
                [str(run.segment_metrics_path), int(horizon_s)],
            ).fetchall()
        return tuple(float(row[0]) for row in rows)

    def available_metric_keys(self) -> tuple[str, ...]:
        available: set[str] = set()
        for run in self.runs():
            rows = self.horizon_metrics(run.run_id)
            if rows:
                available.update(rows[0])
        return tuple(
            metric.key for metric in EVALUATION_METRICS if metric.key in available
        )

    def _read_run(self, manifest_path: Path) -> EvaluationRun | None:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        outputs = payload.get("outputs")
        if not isinstance(outputs, dict):
            return None
        horizon_path = _resolve_artifact_path(
            outputs.get("metrics_by_horizon"), manifest_path.parent
        )
        segment_path = _resolve_artifact_path(
            outputs.get("metrics_by_segment"), manifest_path.parent
        )
        if horizon_path is None or segment_path is None:
            return None
        if not horizon_path.is_file() or not segment_path.is_file():
            return None

        configuration = payload.get("configuration") or {}
        statistics = payload.get("statistics") or {}
        version = str(
            payload.get("version")
            or configuration.get("version")
            or manifest_path.parent.name
        )
        model_type = str(
            payload.get("model_type")
            or configuration.get("model_type")
            or _model_type_from_version(version)
        )
        subsets = configuration.get("subsets") or payload.get("subsets") or []
        horizons = (
            configuration.get("prediction_horizons_seconds")
            or payload.get("prediction_horizons_seconds")
            or []
        )
        split = str(configuration.get("split") or payload.get("split") or "unknown")
        try:
            relative_manifest = manifest_path.relative_to(self.output_root).as_posix()
        except ValueError:
            relative_manifest = manifest_path.as_posix()
        label = str(payload.get("display_name") or _display_name(model_type))
        if version and version.lower() != model_type.lower():
            label = f"{label} · {version}"
        return EvaluationRun(
            run_id=relative_manifest,
            label=label,
            model_type=model_type,
            version=version,
            split=split,
            subset_ids=tuple(int(value) for value in subsets),
            horizons=tuple(int(value) for value in horizons),
            manifest_path=manifest_path.resolve(),
            horizon_metrics_path=horizon_path.resolve(),
            segment_metrics_path=segment_path.resolve(),
            evaluated_segments=int(statistics.get("evaluated_segments") or 0),
            evaluation_protocol=str(
                payload.get("evaluation_protocol")
                or configuration.get("evaluation_protocol")
                or "available targets by horizon"
            ),
        )


def _resolve_artifact_path(value: object, directory: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    local_path = directory / path.name
    return local_path if local_path.is_file() else path


def _convert_metric_row(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    integer_fields = {
        "horizon_s",
        "candidate_origins",
        "evaluated_targets",
        "barometric_evaluated_targets",
    }
    for key, value in row.items():
        if value in {"", None}:
            converted[key] = None
        elif key in integer_fields:
            converted[key] = int(value)
        else:
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
    return converted


def _model_type_from_version(version: str) -> str:
    parts = version.rsplit("_v", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else version


def _display_name(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()
