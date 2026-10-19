"""Configuration for causal IMM trajectory prediction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ImmKalmanConfig:
    version: str
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
    alpha_min: float
    alpha_max: float
    sigma_min: float
    sigma_max: float
    process_q_min: float
    process_q_max: float
    ct_turn_rate_max: float
    model_stickiness_min: float
    model_stickiness_max: float
    motion_gate_enabled: bool
    cv_fallback_history_points: int
    maneuver_gate_block_points: int
    maneuver_gate_min_speed_m_s: float
    maneuver_gate_min_turn_rate_rad_s: float
    maneuver_gate_min_acceleration_m_s2: float
    maneuver_gate_consistency_ratio: float
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
            "model_type": "imm_kalman",
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "split": self.split,
            "trusted_values": list(self.trusted_values),
            "history_seconds": self.history_seconds,
            "history_points": self.history_seconds,
            "prediction_horizons_seconds": list(
                self.prediction_horizons_seconds
            ),
            "origin_stride_seconds": self.origin_stride_seconds,
            "minimum_history_points": self.minimum_history_points,
            "minimum_history_span_seconds": self.minimum_history_span_seconds,
            "skip_interpolated_history_updates": (
                self.skip_interpolated_history_updates
            ),
            "skip_suspicious_history_updates": (
                self.skip_suspicious_history_updates
            ),
            "exclude_interpolated_origins": self.exclude_interpolated_origins,
            "exclude_suspicious_origins": self.exclude_suspicious_origins,
            "exclude_interpolated_targets": self.exclude_interpolated_targets,
            "exclude_suspicious_targets": self.exclude_suspicious_targets,
            "measurement_std_e_m": self.measurement_std_e_m,
            "measurement_std_n_m": self.measurement_std_n_m,
            "measurement_std_u_m": self.measurement_std_u_m,
            "initial_covariance_scale": self.initial_covariance_scale,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "process_q_min": self.process_q_min,
            "process_q_max": self.process_q_max,
            "ct_turn_rate_max": self.ct_turn_rate_max,
            "model_stickiness_min": self.model_stickiness_min,
            "model_stickiness_max": self.model_stickiness_max,
            "motion_gate_enabled": self.motion_gate_enabled,
            "cv_fallback_history_points": self.cv_fallback_history_points,
            "maneuver_gate_block_points": self.maneuver_gate_block_points,
            "maneuver_gate_min_speed_m_s": self.maneuver_gate_min_speed_m_s,
            "maneuver_gate_min_turn_rate_rad_s": (
                self.maneuver_gate_min_turn_rate_rad_s
            ),
            "maneuver_gate_min_acceleration_m_s2": (
                self.maneuver_gate_min_acceleration_m_s2
            ),
            "maneuver_gate_consistency_ratio": (
                self.maneuver_gate_consistency_ratio
            ),
            "prediction_batch_size": self.prediction_batch_size,
            "parquet_batch_rows": self.parquet_batch_rows,
            "prediction_mode": "causal_open_loop",
            "coordinate_system": "forecast_origin_local_enu_approximation",
            "insufficient_history_fallback": "constant_velocity",
            "output_root": self.output_root.as_posix(),
            "config_hash": self.config_hash,
        }


def load_imm_kalman_config(path: Path) -> ImmKalmanConfig:
    """Load a JSON-compatible YAML configuration."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("IMM-Kalman configuration must be an object.")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config = ImmKalmanConfig(
        version=str(_required(payload, "version")),
        dataset_root=Path(_required(payload, "dataset_root")),
        subsets=tuple(int(value) for value in _required(payload, "subsets")),
        split=str(_required(payload, "split")),
        trusted_values=tuple(
            str(value).lower() for value in payload.get("trusted_values", [])
        ),
        history_seconds=int(_required(payload, "history_seconds")),
        prediction_horizons_seconds=tuple(
            int(value)
            for value in _required(payload, "prediction_horizons_seconds")
        ),
        origin_stride_seconds=int(_required(payload, "origin_stride_seconds")),
        minimum_history_points=int(_required(payload, "minimum_history_points")),
        minimum_history_span_seconds=int(
            _required(payload, "minimum_history_span_seconds")
        ),
        skip_interpolated_history_updates=bool(
            _required(payload, "skip_interpolated_history_updates")
        ),
        skip_suspicious_history_updates=bool(
            _required(payload, "skip_suspicious_history_updates")
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
        measurement_std_e_m=float(_required(payload, "measurement_std_e_m")),
        measurement_std_n_m=float(_required(payload, "measurement_std_n_m")),
        measurement_std_u_m=float(_required(payload, "measurement_std_u_m")),
        initial_covariance_scale=float(
            _required(payload, "initial_covariance_scale")
        ),
        alpha_min=float(_required(payload, "alpha_min")),
        alpha_max=float(_required(payload, "alpha_max")),
        sigma_min=float(_required(payload, "sigma_min")),
        sigma_max=float(_required(payload, "sigma_max")),
        process_q_min=float(_required(payload, "process_q_min")),
        process_q_max=float(_required(payload, "process_q_max")),
        ct_turn_rate_max=float(_required(payload, "ct_turn_rate_max")),
        model_stickiness_min=float(_required(payload, "model_stickiness_min")),
        model_stickiness_max=float(_required(payload, "model_stickiness_max")),
        motion_gate_enabled=bool(payload.get("motion_gate_enabled", False)),
        cv_fallback_history_points=int(
            payload.get("cv_fallback_history_points", 31)
        ),
        maneuver_gate_block_points=int(
            payload.get("maneuver_gate_block_points", 11)
        ),
        maneuver_gate_min_speed_m_s=float(
            payload.get("maneuver_gate_min_speed_m_s", 50.0)
        ),
        maneuver_gate_min_turn_rate_rad_s=float(
            payload.get("maneuver_gate_min_turn_rate_rad_s", 0.002)
        ),
        maneuver_gate_min_acceleration_m_s2=float(
            payload.get("maneuver_gate_min_acceleration_m_s2", 0.3)
        ),
        maneuver_gate_consistency_ratio=float(
            payload.get("maneuver_gate_consistency_ratio", 0.5)
        ),
        prediction_batch_size=int(_required(payload, "prediction_batch_size")),
        parquet_batch_rows=int(_required(payload, "parquet_batch_rows")),
        output_root=Path(_required(payload, "output_root")),
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    _validate_config(config)
    return config


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing IMM-Kalman configuration key: {key}")
    return mapping[key]


def _validate_config(config: ImmKalmanConfig) -> None:
    if not config.version.strip():
        raise ValueError("version must not be empty.")
    if not config.subsets or any(value < 1 for value in config.subsets):
        raise ValueError("subsets must contain positive integers.")
    if config.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if any(value not in {"true", "false", "unknown"} for value in config.trusted_values):
        raise ValueError("trusted_values must contain true, false, or unknown.")
    if config.history_seconds < 4:
        raise ValueError("history_seconds must be at least four.")
    horizons = config.prediction_horizons_seconds
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("prediction_horizons_seconds must contain positive integers.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError(
            "prediction_horizons_seconds must be unique and strictly increasing."
        )
    if config.origin_stride_seconds < 1:
        raise ValueError("origin_stride_seconds must be positive.")
    if not 3 <= config.minimum_history_points <= config.history_seconds:
        raise ValueError(
            "minimum_history_points must be between three and history_seconds."
        )
    if not 1 <= config.minimum_history_span_seconds < config.history_seconds:
        raise ValueError(
            "minimum_history_span_seconds must be positive and less than history_seconds."
        )
    if min(
        config.measurement_std_e_m,
        config.measurement_std_n_m,
        config.measurement_std_u_m,
        config.initial_covariance_scale,
    ) <= 0:
        raise ValueError("measurement standard deviations and covariance must be positive.")
    if not 0 < config.alpha_min <= config.alpha_max:
        raise ValueError("alpha bounds are invalid.")
    if not 0 < config.sigma_min <= config.sigma_max:
        raise ValueError("sigma bounds are invalid.")
    if not 0 < config.process_q_min <= config.process_q_max:
        raise ValueError("process-noise bounds are invalid.")
    if config.ct_turn_rate_max <= 0:
        raise ValueError("ct_turn_rate_max must be positive.")
    if not 0 < config.model_stickiness_min <= config.model_stickiness_max < 1:
        raise ValueError("model-stickiness bounds are invalid.")
    if not 3 <= config.cv_fallback_history_points <= config.history_seconds:
        raise ValueError(
            "cv_fallback_history_points must be between three and history_seconds."
        )
    if config.maneuver_gate_block_points < 3:
        raise ValueError("maneuver_gate_block_points must be at least three.")
    required_gate_points = 3 * config.maneuver_gate_block_points - 2
    if required_gate_points > config.history_seconds:
        raise ValueError(
            "Three overlapping maneuver-gate blocks must fit in history_seconds."
        )
    if min(
        config.maneuver_gate_min_speed_m_s,
        config.maneuver_gate_min_turn_rate_rad_s,
        config.maneuver_gate_min_acceleration_m_s2,
    ) <= 0:
        raise ValueError("Maneuver-gate thresholds must be positive.")
    if not 0 < config.maneuver_gate_consistency_ratio <= 1:
        raise ValueError("maneuver_gate_consistency_ratio must be in (0, 1].")
    if config.prediction_batch_size < 1 or config.parquet_batch_rows < 1:
        raise ValueError("batch sizes must be positive.")
