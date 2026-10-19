"""Configuration loading for the OpenSky preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SplitConfig:
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    version: str
    dataset_root: Path
    subsets: tuple[int, ...]
    sampling_interval_seconds: int
    macro_gap_seconds: float
    max_interpolation_missing_steps: int
    coordinate_conflict_threshold_m: float
    suspicious_horizontal_speed_mps: float
    strong_anomaly_horizontal_speed_mps: float
    minimum_segment_points: int
    minimum_segment_duration_seconds: float
    parquet_compression: str
    parquet_row_group_rows: int
    duckdb_memory_limit: str
    duckdb_threads: int
    split: SplitConfig
    source_path: Path
    config_hash: str

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "macro_gap_seconds": self.macro_gap_seconds,
            "max_interpolation_missing_steps": self.max_interpolation_missing_steps,
            "coordinate_conflict_threshold_m": self.coordinate_conflict_threshold_m,
            "suspicious_horizontal_speed_mps": self.suspicious_horizontal_speed_mps,
            "strong_anomaly_horizontal_speed_mps": self.strong_anomaly_horizontal_speed_mps,
            "minimum_segment_points": self.minimum_segment_points,
            "minimum_segment_duration_seconds": self.minimum_segment_duration_seconds,
            "parquet_compression": self.parquet_compression,
            "parquet_row_group_rows": self.parquet_row_group_rows,
            "duckdb_memory_limit": self.duckdb_memory_limit,
            "duckdb_threads": self.duckdb_threads,
            "split": {
                "seed": self.split.seed,
                "train_ratio": self.split.train_ratio,
                "validation_ratio": self.split.validation_ratio,
                "test_ratio": self.split.test_ratio,
            },
            "config_hash": self.config_hash,
        }


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing preprocessing configuration key: {key}")
    return mapping[key]


def load_config(path: Path) -> PreprocessConfig:
    """Load JSON-compatible YAML without adding a YAML runtime dependency."""

    resolved = path.resolve()
    try:
        raw_text = resolved.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{resolved} must use JSON-compatible YAML syntax: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Preprocessing configuration must be an object.")

    split_payload = _required(payload, "split")
    if not isinstance(split_payload, dict):
        raise ValueError("split must be an object.")
    split = SplitConfig(
        seed=int(_required(split_payload, "seed")),
        train_ratio=float(_required(split_payload, "train_ratio")),
        validation_ratio=float(_required(split_payload, "validation_ratio")),
        test_ratio=float(_required(split_payload, "test_ratio")),
    )
    if not math.isclose(
        split.train_ratio + split.validation_ratio + split.test_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    if min(split.train_ratio, split.validation_ratio, split.test_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")

    subsets = tuple(int(value) for value in _required(payload, "subsets"))
    if not subsets or any(value < 1 for value in subsets):
        raise ValueError("subsets must contain positive integers.")

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    config = PreprocessConfig(
        version=str(_required(payload, "version")),
        dataset_root=Path(_required(payload, "dataset_root")),
        subsets=subsets,
        sampling_interval_seconds=int(_required(payload, "sampling_interval_seconds")),
        macro_gap_seconds=float(_required(payload, "macro_gap_seconds")),
        max_interpolation_missing_steps=int(
            _required(payload, "max_interpolation_missing_steps")
        ),
        coordinate_conflict_threshold_m=float(
            _required(payload, "coordinate_conflict_threshold_m")
        ),
        suspicious_horizontal_speed_mps=float(
            _required(payload, "suspicious_horizontal_speed_mps")
        ),
        strong_anomaly_horizontal_speed_mps=float(
            _required(payload, "strong_anomaly_horizontal_speed_mps")
        ),
        minimum_segment_points=int(_required(payload, "minimum_segment_points")),
        minimum_segment_duration_seconds=float(
            payload.get("minimum_segment_duration_seconds", 0.0)
        ),
        parquet_compression=str(_required(payload, "parquet_compression")),
        parquet_row_group_rows=int(_required(payload, "parquet_row_group_rows")),
        duckdb_memory_limit=str(_required(payload, "duckdb_memory_limit")),
        duckdb_threads=int(_required(payload, "duckdb_threads")),
        split=split,
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    if config.sampling_interval_seconds < 1:
        raise ValueError("sampling_interval_seconds must be at least 1.")
    if config.macro_gap_seconds <= config.sampling_interval_seconds:
        raise ValueError("macro_gap_seconds must exceed the sampling interval.")
    if config.max_interpolation_missing_steps < 0:
        raise ValueError("max_interpolation_missing_steps must be non-negative.")
    if config.coordinate_conflict_threshold_m <= 0:
        raise ValueError("coordinate_conflict_threshold_m must be positive.")
    if not (
        0
        < config.suspicious_horizontal_speed_mps
        < config.strong_anomaly_horizontal_speed_mps
    ):
        raise ValueError("Horizontal speed thresholds are inconsistent.")
    if config.minimum_segment_points < 2:
        raise ValueError("minimum_segment_points must be at least 2.")
    if config.minimum_segment_duration_seconds < 0:
        raise ValueError("minimum_segment_duration_seconds must be non-negative.")
    if config.parquet_row_group_rows < 1:
        raise ValueError("parquet_row_group_rows must be positive.")
    if not config.duckdb_memory_limit.strip():
        raise ValueError("duckdb_memory_limit must not be empty.")
    if config.duckdb_threads < 1:
        raise ValueError("duckdb_threads must be positive.")
    return config
