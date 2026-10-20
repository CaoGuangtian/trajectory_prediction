"""Trainable residual correction on top of frozen Standard-IMM forecasts."""

from __future__ import annotations

import torch
from torch import nn

from trajectory_prediction.models.patchtst.model import TrajectoryPatchTST

from .config import ImmPatchTSTConfig


class ImmPatchTSTResidual(nn.Module):
    """Add a PatchTST residual to a non-trainable normalized IMM forecast."""

    def __init__(
        self,
        config: ImmPatchTSTConfig,
        residual_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.residual_model = residual_model or TrajectoryPatchTST(config.patchtst)

    def predict_residual(self, history: torch.Tensor) -> torch.Tensor:
        return self.residual_model(history)

    def forward(
        self,
        history: torch.Tensor,
        imm_prediction: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            self.config.patchtst.prediction_points,
            TrajectoryPatchTST.channels,
        )
        if imm_prediction.ndim != 3 or tuple(imm_prediction.shape[1:]) != expected:
            raise ValueError(
                f"Expected IMM prediction shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(imm_prediction.shape)}."
            )
        residual = self.predict_residual(history)
        if residual.shape != imm_prediction.shape:
            raise ValueError("PatchTST residual and IMM prediction shapes must match.")
        return imm_prediction + residual


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
