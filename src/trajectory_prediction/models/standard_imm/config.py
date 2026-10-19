"""Configuration for fixed-parameter standard IMM and CV Kalman models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_MODELS = ("cv", "ct_left", "ct_right", "singer")


@dataclass(frozen=True, slots=True)
class StandardImmConfig:
    version: str
    implementation_version: str
    model_type: str
    display_name: str
    dataset_root: Path
    subsets: tuple[int, ...]
    split: str
    trusted_values: tuple[str, ...]
    history_seconds: int
    prediction_horizons_seconds: tuple[int, ...]
    origin_stride_seconds: int
    minimum_history_points: int
    minimum_history_span_seconds: int
    skip_interpolated_history_updates: bool
    skip_suspicious_history_updates: bool
    exclude_interpolated_origins: bool
    exclude_suspicious_origins: bool
    exclude_interpolated_targets: bool
    exclude_suspicious_targets: bool
    measurement_std_e_m: float
    measurement_std_n_m: float
    measurement_std_u_m: float
    initial_covariance_scale: float
    model_names: tuple[str, ...]
    initial_model_probabilities: tuple[float, ...]
    model_transition_matrix: tuple[tuple[float, ...], ...]
    cv_process_q_xyz: tuple[float, float, float]
    ct_process_q_xyz: tuple[float, float, float]
    ct_turn_rate_rad_s: float
    singer_alpha_xyz: tuple[float, float, float]
    singer_sigma_xyz: tuple[float, float, float]
    prediction_batch_size: int
    parquet_batch_rows: int
    output_root: Path
    source_path: Path
    config_hash: str

    @property
    def measurement_variances(self) -> tuple[float, float, float]:
        return (
            self.measurement_std_e_m**2,
            self.measurement_std_n_m**2,
            self.measurement_std_u_m**2,
        )

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "implementation_version": self.implementation_version,
            "model_type": self.model_type,
            "display_name": self.display_name,
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "split": self.split,
            "trusted_values": list(self.trusted_values),
            "history_seconds": self.history_seconds,
            "history_points": self.history_seconds,
            "prediction_horizons_seconds": list(self.prediction_horizons_seconds),
            "origin_stride_seconds": self.origin_stride_seconds,
            "minimum_history_points": self.minimum_history_points,
            "minimum_history_span_seconds": self.minimum_history_span_seconds,
            "skip_interpolated_history_updates": self.skip_interpolated_history_updates,
            "skip_suspicious_history_updates": self.skip_suspicious_history_updates,
            "exclude_interpolated_origins": self.exclude_interpolated_origins,
            "exclude_suspicious_origins": self.exclude_suspicious_origins,
            "exclude_interpolated_targets": self.exclude_interpolated_targets,
            "exclude_suspicious_targets": self.exclude_suspicious_targets,
            "measurement_std_e_m": self.measurement_std_e_m,
            "measurement_std_n_m": self.measurement_std_n_m,
            "measurement_std_u_m": self.measurement_std_u_m,
            "initial_covariance_scale": self.initial_covariance_scale,
            "model_names": list(self.model_names),
            "initial_model_probabilities": list(self.initial_model_probabilities),
            "model_transition_matrix": [list(row) for row in self.model_transition_matrix],
            "cv_process_q_xyz": list(self.cv_process_q_xyz),
            "ct_process_q_xyz": list(self.ct_process_q_xyz),
            "ct_turn_rate_rad_s": self.ct_turn_rate_rad_s,
            "singer_alpha_xyz": list(self.singer_alpha_xyz),
            "singer_sigma_xyz": list(self.singer_sigma_xyz),
            "prediction_batch_size": self.prediction_batch_size,
            "parquet_batch_rows": self.parquet_batch_rows,
            "prediction_mode": "causal_open_loop",
            "parameter_policy": "fixed_from_configuration",
            "coordinate_system": "forecast_origin_local_enu_approximation",
            "insufficient_history_fallback": "position_hold",
            "output_root": self.output_root.as_posix(),
            "config_hash": self.config_hash,
        }


def load_standard_imm_config(path: Path) -> StandardImmConfig:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Standard IMM configuration must be an object.")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def required(key: str) -> Any:
        if key not in payload:
            raise ValueError(f"Missing Standard IMM configuration key: {key}")
        return payload[key]

    def xyz(key: str) -> tuple[float, float, float]:
        values = tuple(float(value) for value in required(key))
        if len(values) != 3:
            raise ValueError(f"{key} must contain exactly three values.")
        return values

    config = StandardImmConfig(
        version=str(required("version")),
        implementation_version=str(required("implementation_version")),
        model_type=str(required("model_type")),
        display_name=str(required("display_name")),
        dataset_root=Path(required("dataset_root")),
        subsets=tuple(int(value) for value in required("subsets")),
        split=str(required("split")),
        trusted_values=tuple(str(value).lower() for value in payload.get("trusted_values", [])),
        history_seconds=int(required("history_seconds")),
        prediction_horizons_seconds=tuple(int(value) for value in required("prediction_horizons_seconds")),
        origin_stride_seconds=int(required("origin_stride_seconds")),
        minimum_history_points=int(required("minimum_history_points")),
        minimum_history_span_seconds=int(required("minimum_history_span_seconds")),
        skip_interpolated_history_updates=bool(required("skip_interpolated_history_updates")),
        skip_suspicious_history_updates=bool(required("skip_suspicious_history_updates")),
        exclude_interpolated_origins=bool(required("exclude_interpolated_origins")),
        exclude_suspicious_origins=bool(required("exclude_suspicious_origins")),
        exclude_interpolated_targets=bool(required("exclude_interpolated_targets")),
        exclude_suspicious_targets=bool(required("exclude_suspicious_targets")),
        measurement_std_e_m=float(required("measurement_std_e_m")),
        measurement_std_n_m=float(required("measurement_std_n_m")),
        measurement_std_u_m=float(required("measurement_std_u_m")),
        initial_covariance_scale=float(required("initial_covariance_scale")),
        model_names=tuple(str(value) for value in required("model_names")),
        initial_model_probabilities=tuple(float(value) for value in required("initial_model_probabilities")),
        model_transition_matrix=tuple(tuple(float(value) for value in row) for row in required("model_transition_matrix")),
        cv_process_q_xyz=xyz("cv_process_q_xyz"),
        ct_process_q_xyz=xyz("ct_process_q_xyz"),
        ct_turn_rate_rad_s=float(required("ct_turn_rate_rad_s")),
        singer_alpha_xyz=xyz("singer_alpha_xyz"),
        singer_sigma_xyz=xyz("singer_sigma_xyz"),
        prediction_batch_size=int(required("prediction_batch_size")),
        parquet_batch_rows=int(required("parquet_batch_rows")),
        output_root=Path(required("output_root")),
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    _validate(config)
    return config


def _validate(config: StandardImmConfig) -> None:
    if not all((config.version.strip(), config.implementation_version.strip(), config.model_type.strip(), config.display_name.strip())):
        raise ValueError("Version and model identity fields must not be empty.")
    if not config.subsets or any(value < 1 for value in config.subsets):
        raise ValueError("subsets must contain positive integers.")
    if config.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if any(value not in {"true", "false", "unknown"} for value in config.trusted_values):
        raise ValueError("trusted_values must contain true, false, or unknown.")
    if config.history_seconds < 4:
        raise ValueError("history_seconds must be at least four.")
    horizons = config.prediction_horizons_seconds
    if not horizons or tuple(sorted(set(horizons))) != horizons or min(horizons) < 1:
        raise ValueError("prediction_horizons_seconds must be positive, unique, and increasing.")
    if config.origin_stride_seconds < 1:
        raise ValueError("origin_stride_seconds must be positive.")
    if not 3 <= config.minimum_history_points <= config.history_seconds:
        raise ValueError("minimum_history_points is outside the history window.")
    if not 1 <= config.minimum_history_span_seconds < config.history_seconds:
        raise ValueError("minimum_history_span_seconds is outside the history window.")
    if min(config.measurement_std_e_m, config.measurement_std_n_m, config.measurement_std_u_m, config.initial_covariance_scale) <= 0:
        raise ValueError("Measurement standard deviations and covariance must be positive.")
    if not config.model_names or len(set(config.model_names)) != len(config.model_names):
        raise ValueError("model_names must be nonempty and unique.")
    if any(name not in SUPPORTED_MODELS for name in config.model_names):
        raise ValueError(f"model_names must be selected from {SUPPORTED_MODELS}.")
    count = len(config.model_names)
    if len(config.initial_model_probabilities) != count or len(config.model_transition_matrix) != count:
        raise ValueError("Model probabilities and transition matrix must match model_names.")
    if any(len(row) != count for row in config.model_transition_matrix):
        raise ValueError("model_transition_matrix must be square.")
    if min(config.initial_model_probabilities) < 0 or abs(sum(config.initial_model_probabilities) - 1.0) > 1.0e-8:
        raise ValueError("initial_model_probabilities must be nonnegative and sum to one.")
    if any(min(row) < 0 or abs(sum(row) - 1.0) > 1.0e-8 for row in config.model_transition_matrix):
        raise ValueError("Each model_transition_matrix row must be nonnegative and sum to one.")
    if min(*config.cv_process_q_xyz, *config.ct_process_q_xyz, *config.singer_alpha_xyz, *config.singer_sigma_xyz) <= 0:
        raise ValueError("Process parameters must be positive.")
    if config.ct_turn_rate_rad_s <= 0:
        raise ValueError("ct_turn_rate_rad_s must be positive.")
    if config.prediction_batch_size < 1 or config.parquet_batch_rows < 1:
        raise ValueError("Batch sizes must be positive.")
