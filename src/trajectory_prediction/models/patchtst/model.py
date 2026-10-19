"""Project wrapper around the reference-aligned PatchTST backbone."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import ReferenceAlignedPatchTSTBackbone
from .config import PatchTSTConfig


class TrajectoryPatchTST(nn.Module):
    """Directly predict future local-ENU offsets from a causal history."""

    channels = 3

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = ReferenceAlignedPatchTSTBackbone(config)
        self.patch_count = self.backbone.patch_count

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.config.history_points, self.channels)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                f"Expected history shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(history.shape)}."
            )
        return self.backbone(history.transpose(1, 2)).transpose(1, 2)


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
