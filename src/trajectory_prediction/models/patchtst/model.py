"""A compact channel-independent PatchTST for local ENU trajectories."""

from __future__ import annotations

import torch
from torch import nn

from .config import PatchTSTConfig


class ReversibleInstanceNorm(nn.Module):
    def __init__(self, channels: int, affine: bool, subtract_last: bool) -> None:
        super().__init__()
        self.subtract_last = subtract_last
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        center = values[:, -1:, :] if self.subtract_last else values.mean(1, keepdim=True)
        scale = torch.sqrt(values.var(1, keepdim=True, unbiased=False) + 1.0e-5)
        normalized = (values - center) / scale
        if self.weight is not None:
            normalized = normalized * self.weight + self.bias
        return normalized, (center, scale)

    def denormalize(
        self,
        values: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        center, scale = state
        if self.weight is not None:
            values = (values - self.bias) / (self.weight + 1.0e-8)
        return values * scale + center


class EncoderBlock(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        dimension = config.model_dimension
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            config.attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, config.feedforward_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dimension, dimension),
            nn.Dropout(config.dropout),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        values = values + self.residual_dropout(attended)
        return values + self.feedforward(self.norm2(values))


class TrajectoryPatchTST(nn.Module):
    """Directly predicts all future ENU offsets from a causal history window."""

    channels = 3

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        effective_length = config.history_points + (
            config.patch_stride if config.patch_padding_end else 0
        )
        self.patch_count = (
            (effective_length - config.patch_length) // config.patch_stride + 1
        )
        if self.patch_count < 1:
            raise ValueError("Patch configuration produces no patches.")
        self.revin = (
            ReversibleInstanceNorm(
                self.channels,
                affine=config.revin_affine,
                subtract_last=config.revin_subtract_last,
            )
            if config.revin_enabled
            else None
        )
        self.patch_projection = nn.Linear(config.patch_length, config.model_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_count, config.model_dimension)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.encoder = nn.Sequential(
            *(EncoderBlock(config) for _ in range(config.encoder_layers))
        )
        self.final_norm = nn.LayerNorm(config.model_dimension)
        head_input = self.patch_count * config.model_dimension
        if config.individual_heads:
            self.heads = nn.ModuleList(
                nn.Sequential(
                    nn.Dropout(config.head_dropout),
                    nn.Linear(head_input, config.prediction_points),
                )
                for _ in range(self.channels)
            )
            self.shared_head = None
        else:
            self.heads = None
            self.shared_head = nn.Sequential(
                nn.Dropout(config.head_dropout),
                nn.Linear(head_input, config.prediction_points),
            )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.config.history_points, self.channels)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                f"Expected history shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(history.shape)}."
            )
        revin_state = None
        if self.revin is not None:
            history, revin_state = self.revin.normalize(history)
        values = history.transpose(1, 2)
        if self.config.patch_padding_end:
            values = nn.functional.pad(
                values, (0, self.config.patch_stride), mode="replicate"
            )
        patches = values.unfold(
            dimension=-1,
            size=self.config.patch_length,
            step=self.config.patch_stride,
        )
        batch, channels, patch_count, patch_length = patches.shape
        encoded = self.patch_projection(
            patches.reshape(batch * channels, patch_count, patch_length)
        )
        encoded = encoded + self.position_embedding
        encoded = self.final_norm(self.encoder(encoded))
        flattened = encoded.reshape(batch, channels, -1)
        if self.heads is not None:
            prediction = torch.stack(
                [self.heads[channel](flattened[:, channel]) for channel in range(channels)],
                dim=2,
            )
        else:
            prediction = self.shared_head(flattened).transpose(1, 2)
        if self.revin is not None and revin_state is not None:
            prediction = self.revin.denormalize(prediction, revin_state)
        return prediction


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
