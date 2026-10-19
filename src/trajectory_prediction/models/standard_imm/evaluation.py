"""Shared-protocol evaluation entry point for the standard IMM family."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from trajectory_prediction.models.imm_kalman.evaluation import (
    ImmKalmanSource,
    evaluate_imm_model,
)

from .config import StandardImmConfig
from .core import predict_standard_imm_batch


StandardImmSource = ImmKalmanSource


def evaluate_standard_imm(
    sources: Iterable[StandardImmSource],
    config: StandardImmConfig,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    return evaluate_imm_model(
        sources,
        config,
        output_directory,
        predictor=predict_standard_imm_batch,
        model_names=config.model_names,
        artifact=f"{config.model_type}_evaluation",
        overwrite=overwrite,
    )
