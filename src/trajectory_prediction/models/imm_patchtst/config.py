"""Configuration composition for IMM-PatchTST residual fusion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectory_prediction.models.patchtst.config import (
    PatchTSTConfig,
    load_patchtst_config,
)
from trajectory_prediction.models.standard_imm.config import (
    StandardImmConfig,
    load_standard_imm_config,
)


@dataclass(frozen=True, slots=True)
class ImmPatchTSTConfig:
    version: str
    implementation_version: str
    model_type: str
    display_name: str
    fusion_mode: str
    patchtst_config_path: Path
    imm_config_path: Path
    cache_root: Path
    cache_batch_size: int
    output_root: Path
    patchtst: PatchTSTConfig
    imm: StandardImmConfig
    source_path: Path
    config_hash: str

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "implementation_version": self.implementation_version,
            "model_type": self.model_type,
            "display_name": self.display_name,
            "fusion_mode": self.fusion_mode,
            "patchtst_config_path": self.patchtst_config_path.as_posix(),
            "imm_config_path": self.imm_config_path.as_posix(),
            "cache_root": self.cache_root.as_posix(),
            "cache_batch_size": self.cache_batch_size,
            "output_root": self.output_root.as_posix(),
            "subsets": list(self.patchtst.subsets),
            "history_points": self.patchtst.history_points,
            "prediction_points": self.patchtst.prediction_points,
            "prediction_horizons_seconds": list(
                self.patchtst.prediction_horizons_seconds
            ),
            "train_origin_stride_seconds": (
                self.patchtst.train_origin_stride_seconds
            ),
            "validation_origin_stride_seconds": (
                self.patchtst.validation_origin_stride_seconds
            ),
            "test_origin_stride_seconds": self.patchtst.test_origin_stride_seconds,
            "patchtst_config_hash": self.patchtst.config_hash,
            "imm_config_hash": self.imm.config_hash,
            "config_hash": self.config_hash,
            "coordinate_system": "forecast_origin_local_enu_approximation",
            "prediction_mode": "causal_imm_plus_direct_residual",
        }


def load_imm_patchtst_config(path: Path) -> ImmPatchTSTConfig:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("IMM-PatchTST configuration must be an object.")

    def required(key: str) -> Any:
        if key not in payload:
            raise ValueError(f"Missing IMM-PatchTST configuration key: {key}")
        return payload[key]

    patchtst_path = _resolve_reference(str(required("patchtst_config")), resolved)
    imm_path = _resolve_reference(str(required("imm_config")), resolved)
    patchtst = load_patchtst_config(patchtst_path)
    imm = load_standard_imm_config(imm_path)
    hash_payload = {
        **payload,
        "patchtst_config_hash": patchtst.config_hash,
        "imm_config_hash": imm.config_hash,
    }
    canonical = json.dumps(
        hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    config = ImmPatchTSTConfig(
        version=str(required("version")),
        implementation_version=str(required("implementation_version")),
        model_type=str(required("model_type")),
        display_name=str(required("display_name")),
        fusion_mode=str(required("fusion_mode")).lower(),
        patchtst_config_path=patchtst_path,
        imm_config_path=imm_path,
        cache_root=Path(required("cache_root")),
        cache_batch_size=int(required("cache_batch_size")),
        output_root=Path(required("output_root")),
        patchtst=patchtst,
        imm=imm,
        source_path=resolved,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )
    _validate(config)
    return config


def _resolve_reference(value: str, owner: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    working_directory_path = path.resolve()
    if working_directory_path.is_file():
        return working_directory_path
    return (owner.parent / path).resolve()


def _validate(config: ImmPatchTSTConfig) -> None:
    if not all(
        (
            config.version.strip(),
            config.implementation_version.strip(),
            config.model_type.strip(),
            config.display_name.strip(),
        )
    ):
        raise ValueError("Fusion model identity fields must not be empty.")
    if config.fusion_mode != "residual":
        raise ValueError("The first IMM-PatchTST implementation supports residual fusion only.")
    if config.cache_batch_size < 1:
        raise ValueError("cache_batch_size must be positive.")
    patchtst = config.patchtst
    imm = config.imm
    if patchtst.model_type != "patchtst" or patchtst.input_transform != "none":
        raise ValueError(
            "The v1 residual fusion requires the unsmoothed PatchTST Raw branch."
        )
    if imm.model_type != "standard_imm" or len(imm.model_names) < 2:
        raise ValueError(
            "The v1 residual fusion requires a multi-model Standard IMM branch."
        )
    if patchtst.history_points != imm.history_seconds:
        raise ValueError("PatchTST and IMM history lengths must match.")
    if patchtst.dataset_root.resolve() != imm.dataset_root.resolve():
        raise ValueError("PatchTST and IMM must use the same dataset root.")
    if patchtst.subsets != imm.subsets:
        raise ValueError("PatchTST and IMM configured subsets must match.")
    if patchtst.trusted_values != imm.trusted_values:
        raise ValueError("PatchTST and IMM trusted-value filters must match.")
    quality_pairs = (
        (patchtst.exclude_interpolated_origins, imm.exclude_interpolated_origins),
        (patchtst.exclude_suspicious_origins, imm.exclude_suspicious_origins),
        (patchtst.exclude_interpolated_targets, imm.exclude_interpolated_targets),
        (patchtst.exclude_suspicious_targets, imm.exclude_suspicious_targets),
    )
    if any(left != right for left, right in quality_pairs):
        raise ValueError(
            "PatchTST and IMM origin/target quality policies must match."
        )
