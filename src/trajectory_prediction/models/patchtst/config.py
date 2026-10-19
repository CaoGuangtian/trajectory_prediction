"""Configuration for the causal PatchTST trajectory baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PatchTSTConfig:
    version: str
    implementation_version: str
    model_type: str
    display_name: str
    dataset_root: Path
    subsets: tuple[int, ...]
    trusted_values: tuple[str, ...]
    history_points: int
    prediction_points: int
    prediction_horizons_seconds: tuple[int, ...]
    train_origin_stride_seconds: int
    validation_origin_stride_seconds: int
    test_origin_stride_seconds: int
    exclude_interpolated_origins: bool
    exclude_suspicious_origins: bool
    exclude_interpolated_targets: bool
    exclude_suspicious_targets: bool
    normalization_max_origins: int
    normalization_min_scale_m: float
    input_transform: str
    smooth_channels: tuple[str, ...]
    ema_alpha: float
    smooth_recenter_at_origin: bool
    patch_length: int
    patch_stride: int
    patch_padding_end: bool
    model_dimension: int
    attention_heads: int
    encoder_layers: int
    feedforward_dimension: int
    dropout: float
    attention_dropout: float
    head_dropout: float
    individual_heads: bool
    encoder_normalization: str
    residual_attention: bool
    pre_norm: bool
    positional_encoding: str
    learn_positional_encoding: bool
    revin_enabled: bool
    revin_affine: bool
    revin_subtract_last: bool
    batch_size: int
    evaluation_batch_size: int
    learning_rate: float
    weight_decay: float
    maximum_epochs: int
    early_stopping_patience: int
    huber_delta: float
    random_seed: int
    device: str
    cuda_device_index: int
    cpu_threads: int
    dataloader_workers: int
    use_mixed_precision: bool
    output_root: Path
    source_path: Path
    config_hash: str

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "implementation_version": self.implementation_version,
            "model_type": self.model_type,
            "display_name": self.display_name,
            "dataset_root": self.dataset_root.as_posix(),
            "subsets": list(self.subsets),
            "trusted_values": list(self.trusted_values),
            "history_points": self.history_points,
            "prediction_points": self.prediction_points,
            "prediction_horizons_seconds": list(self.prediction_horizons_seconds),
            "train_origin_stride_seconds": self.train_origin_stride_seconds,
            "validation_origin_stride_seconds": self.validation_origin_stride_seconds,
            "test_origin_stride_seconds": self.test_origin_stride_seconds,
            "exclude_interpolated_origins": self.exclude_interpolated_origins,
            "exclude_suspicious_origins": self.exclude_suspicious_origins,
            "exclude_interpolated_targets": self.exclude_interpolated_targets,
            "exclude_suspicious_targets": self.exclude_suspicious_targets,
            "normalization_max_origins": self.normalization_max_origins,
            "normalization_min_scale_m": self.normalization_min_scale_m,
            "input_transform": self.input_transform,
            "smooth_channels": list(self.smooth_channels),
            "ema_alpha": self.ema_alpha,
            "smooth_recenter_at_origin": self.smooth_recenter_at_origin,
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "patch_padding_end": self.patch_padding_end,
            "model_dimension": self.model_dimension,
            "attention_heads": self.attention_heads,
            "encoder_layers": self.encoder_layers,
            "feedforward_dimension": self.feedforward_dimension,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "head_dropout": self.head_dropout,
            "individual_heads": self.individual_heads,
            "encoder_normalization": self.encoder_normalization,
            "residual_attention": self.residual_attention,
            "pre_norm": self.pre_norm,
            "positional_encoding": self.positional_encoding,
            "learn_positional_encoding": self.learn_positional_encoding,
            "revin_enabled": self.revin_enabled,
            "revin_affine": self.revin_affine,
            "revin_subtract_last": self.revin_subtract_last,
            "batch_size": self.batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "maximum_epochs": self.maximum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "huber_delta": self.huber_delta,
            "random_seed": self.random_seed,
            "device": self.device,
            "cuda_device_index": self.cuda_device_index,
            "cpu_threads": self.cpu_threads,
            "dataloader_workers": self.dataloader_workers,
            "use_mixed_precision": self.use_mixed_precision,
            "output_root": self.output_root.as_posix(),
            "input_features": ["east_m", "north_m", "up_m"],
            "target_features": ["east_m", "north_m", "up_m"],
            "coordinate_system": "forecast_origin_local_enu_approximation",
            "prediction_mode": "causal_direct_multi_horizon",
            "config_hash": self.config_hash,
        }


def load_patchtst_config(path: Path) -> PatchTSTConfig:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PatchTST configuration must be an object.")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def required(key: str) -> Any:
        if key not in payload:
            raise ValueError(f"Missing PatchTST configuration key: {key}")
        return payload[key]

    config = PatchTSTConfig(
        version=str(required("version")),
        implementation_version=str(required("implementation_version")),
        model_type=str(required("model_type")),
        display_name=str(required("display_name")),
        dataset_root=Path(required("dataset_root")),
        subsets=tuple(int(value) for value in required("subsets")),
        trusted_values=tuple(str(value).lower() for value in payload.get("trusted_values", [])),
        history_points=int(required("history_points")),
        prediction_points=int(required("prediction_points")),
        prediction_horizons_seconds=tuple(int(value) for value in required("prediction_horizons_seconds")),
        train_origin_stride_seconds=int(required("train_origin_stride_seconds")),
        validation_origin_stride_seconds=int(required("validation_origin_stride_seconds")),
        test_origin_stride_seconds=int(required("test_origin_stride_seconds")),
        exclude_interpolated_origins=bool(required("exclude_interpolated_origins")),
        exclude_suspicious_origins=bool(required("exclude_suspicious_origins")),
        exclude_interpolated_targets=bool(required("exclude_interpolated_targets")),
        exclude_suspicious_targets=bool(required("exclude_suspicious_targets")),
        normalization_max_origins=int(required("normalization_max_origins")),
        normalization_min_scale_m=float(required("normalization_min_scale_m")),
        input_transform=str(payload.get("input_transform", "none")).lower(),
        smooth_channels=tuple(
            str(value).lower() for value in payload.get("smooth_channels", [])
        ),
        ema_alpha=float(payload.get("ema_alpha", 0.4)),
        smooth_recenter_at_origin=bool(
            payload.get("smooth_recenter_at_origin", True)
        ),
        patch_length=int(required("patch_length")),
        patch_stride=int(required("patch_stride")),
        patch_padding_end=bool(required("patch_padding_end")),
        model_dimension=int(required("model_dimension")),
        attention_heads=int(required("attention_heads")),
        encoder_layers=int(required("encoder_layers")),
        feedforward_dimension=int(required("feedforward_dimension")),
        dropout=float(required("dropout")),
        attention_dropout=float(required("attention_dropout")),
        head_dropout=float(required("head_dropout")),
        individual_heads=bool(required("individual_heads")),
        encoder_normalization=str(required("encoder_normalization")),
        residual_attention=bool(required("residual_attention")),
        pre_norm=bool(required("pre_norm")),
        positional_encoding=str(required("positional_encoding")),
        learn_positional_encoding=bool(required("learn_positional_encoding")),
        revin_enabled=bool(required("revin_enabled")),
        revin_affine=bool(required("revin_affine")),
        revin_subtract_last=bool(required("revin_subtract_last")),
        batch_size=int(required("batch_size")),
        evaluation_batch_size=int(required("evaluation_batch_size")),
        learning_rate=float(required("learning_rate")),
        weight_decay=float(required("weight_decay")),
        maximum_epochs=int(required("maximum_epochs")),
        early_stopping_patience=int(required("early_stopping_patience")),
        huber_delta=float(required("huber_delta")),
        random_seed=int(required("random_seed")),
        device=str(required("device")).lower(),
        cuda_device_index=int(required("cuda_device_index")),
        cpu_threads=int(required("cpu_threads")),
        dataloader_workers=int(required("dataloader_workers")),
        use_mixed_precision=bool(required("use_mixed_precision")),
        output_root=Path(required("output_root")),
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    _validate(config)
    return config


def _validate(config: PatchTSTConfig) -> None:
    if not all((config.version.strip(), config.implementation_version.strip(), config.model_type.strip(), config.display_name.strip())):
        raise ValueError("Version and model identity fields must not be empty.")
    if not config.subsets or any(value < 1 for value in config.subsets):
        raise ValueError("subsets must contain positive integers.")
    if any(value not in {"true", "false", "unknown"} for value in config.trusted_values):
        raise ValueError("trusted_values must contain true, false, or unknown.")
    if config.history_points < 2 or config.prediction_points < 1:
        raise ValueError("History and prediction lengths are invalid.")
    horizons = config.prediction_horizons_seconds
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Prediction horizons must be unique and increasing.")
    if min(horizons) < 1 or max(horizons) > config.prediction_points:
        raise ValueError("Prediction horizons must fit inside prediction_points.")
    if min(config.train_origin_stride_seconds, config.validation_origin_stride_seconds, config.test_origin_stride_seconds) < 1:
        raise ValueError("Origin strides must be positive.")
    if config.normalization_max_origins < 1 or config.normalization_min_scale_m <= 0:
        raise ValueError("Normalization settings are invalid.")
    if config.input_transform not in {"none", "causal_ema"}:
        raise ValueError("input_transform must be none or causal_ema.")
    valid_channels = {"east", "north", "up"}
    if len(config.smooth_channels) != len(set(config.smooth_channels)) or any(
        channel not in valid_channels for channel in config.smooth_channels
    ):
        raise ValueError(
            "smooth_channels must contain unique east, north, or up values."
        )
    if config.input_transform == "none" and config.smooth_channels:
        raise ValueError("smooth_channels must be empty when input_transform is none.")
    if config.input_transform == "causal_ema":
        if not config.smooth_channels:
            raise ValueError("causal_ema requires at least one smooth channel.")
        if not 0.0 < config.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not config.smooth_recenter_at_origin:
            raise ValueError(
                "causal_ema must recenter at the forecast origin for local ENU input."
            )
    if not 1 <= config.patch_length <= config.history_points or config.patch_stride < 1:
        raise ValueError("Patch length and stride are invalid.")
    if config.model_dimension < 1 or config.attention_heads < 1:
        raise ValueError("Transformer dimensions must be positive.")
    if config.model_dimension % config.attention_heads:
        raise ValueError("model_dimension must be divisible by attention_heads.")
    if min(config.encoder_layers, config.feedforward_dimension) < 1:
        raise ValueError("Encoder depth and feedforward dimension must be positive.")
    if any(not 0 <= value < 1 for value in (config.dropout, config.attention_dropout, config.head_dropout)):
        raise ValueError("Dropout values must be in [0, 1).")
    if config.encoder_normalization.lower() not in {"batchnorm", "layernorm"}:
        raise ValueError("encoder_normalization must be BatchNorm or LayerNorm.")
    if config.positional_encoding.lower() not in {"zeros", "sincos"}:
        raise ValueError("positional_encoding must be zeros or sincos.")
    if min(config.batch_size, config.evaluation_batch_size, config.maximum_epochs, config.early_stopping_patience) < 1:
        raise ValueError("Batch sizes, epochs, and patience must be positive.")
    if min(config.learning_rate, config.huber_delta) <= 0 or config.weight_decay < 0:
        raise ValueError("Optimization settings are invalid.")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    if config.cuda_device_index < 0 or config.cpu_threads < 1 or config.dataloader_workers < 0:
        raise ValueError("Runtime device settings are invalid.")
