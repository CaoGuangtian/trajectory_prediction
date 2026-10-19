"""Reference-aligned PatchTST backbone adapted for trajectory forecasting.

This module is adapted from the official PatchTST supervised backbone
(https://github.com/yuqinie98/PatchTST), licensed under Apache-2.0. It keeps
the original patching, channel-independent encoder, residual attention,
normalization, RevIN, and flatten-head behavior while exposing the project's
typed configuration and tensor interface.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import PatchTSTConfig


class ReversibleInstanceNorm(nn.Module):
    def __init__(self, channels: int, affine: bool, subtract_last: bool) -> None:
        super().__init__()
        self.subtract_last = subtract_last
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        center = (
            values[:, -1:, :]
            if self.subtract_last
            else values.mean(dim=1, keepdim=True)
        ).detach()
        scale = torch.sqrt(
            values.var(dim=1, keepdim=True, unbiased=False) + 1.0e-5
        ).detach()
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
            values = (values - self.bias) / (self.weight + 1.0e-10)
        return values * scale + center


class FlattenHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, input_features: int) -> None:
        super().__init__()
        self.individual = config.individual_heads
        if self.individual:
            self.linears = nn.ModuleList(
                nn.Linear(input_features, config.prediction_points)
                for _ in range(ReferenceAlignedPatchTSTBackbone.channels)
            )
            self.dropouts = nn.ModuleList(
                nn.Dropout(config.head_dropout)
                for _ in range(ReferenceAlignedPatchTSTBackbone.channels)
            )
            self.shared = None
        else:
            self.linears = None
            self.dropouts = None
            self.shared = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(input_features, config.prediction_points),
                nn.Dropout(config.head_dropout),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.individual:
            outputs = []
            for channel, (linear, dropout) in enumerate(
                zip(self.linears, self.dropouts, strict=True)
            ):
                flattened = values[:, channel].flatten(start_dim=-2)
                outputs.append(dropout(linear(flattened)))
            return torch.stack(outputs, dim=1)
        return self.shared(values)


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        attention_dropout: float,
        residual_attention: bool,
    ) -> None:
        super().__init__()
        head_dimension = model_dimension // attention_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dimension**-0.5), requires_grad=False
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_attention = residual_attention

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        previous_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.matmul(queries, keys) * self.scale
        if previous_scores is not None:
            scores = scores + previous_scores
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, values)
        return output, weights, scores


class MultiheadAttention(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        dimension = config.model_dimension
        self.heads = config.attention_heads
        self.head_dimension = dimension // self.heads
        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linear(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)
        self.attention = ScaledDotProductAttention(
            dimension,
            self.heads,
            config.attention_dropout,
            config.residual_attention,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(dimension, dimension), nn.Dropout(config.dropout)
        )

    def forward(
        self,
        values: torch.Tensor,
        previous_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = values.shape[0]
        queries = self.query_projection(values).view(
            batch, -1, self.heads, self.head_dimension
        )
        keys = self.key_projection(values).view(
            batch, -1, self.heads, self.head_dimension
        )
        projected_values = self.value_projection(values).view(
            batch, -1, self.heads, self.head_dimension
        )
        queries = queries.transpose(1, 2)
        keys = keys.permute(0, 2, 3, 1)
        projected_values = projected_values.transpose(1, 2)
        output, weights, scores = self.attention(
            queries, keys, projected_values, previous_scores
        )
        output = output.transpose(1, 2).contiguous().view(
            batch, -1, self.heads * self.head_dimension
        )
        return self.output_projection(output), weights, scores


class SequenceNormalization(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.batch_norm = config.encoder_normalization.lower() == "batchnorm"
        self.normalization = (
            nn.BatchNorm1d(config.model_dimension)
            if self.batch_norm
            else nn.LayerNorm(config.model_dimension)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.batch_norm:
            return self.normalization(values.transpose(1, 2)).transpose(1, 2)
        return self.normalization(values)


class TSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.residual_attention = config.residual_attention
        self.attention = MultiheadAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_norm = SequenceNormalization(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dimension, config.feedforward_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dimension, config.model_dimension),
        )
        self.feedforward_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = SequenceNormalization(config)

    def forward(
        self,
        values: torch.Tensor,
        previous_scores: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_input = self.attention_norm(values) if self.pre_norm else values
        attended, _, scores = self.attention(attention_input, previous_scores)
        values = values + self.attention_dropout(attended)
        if not self.pre_norm:
            values = self.attention_norm(values)

        feedforward_input = self.feedforward_norm(values) if self.pre_norm else values
        values = values + self.feedforward_dropout(
            self.feedforward(feedforward_input)
        )
        if not self.pre_norm:
            values = self.feedforward_norm(values)
        return values, scores if self.residual_attention else None


class TSTEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TSTEncoderLayer(config) for _ in range(config.encoder_layers)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scores = None
        for layer in self.layers:
            values, scores = layer(values, scores)
        return values


class ChannelIndependentEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig, patch_count: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(
            config.patch_length, config.model_dimension
        )
        self.position_embedding = nn.Parameter(
            _position_encoding(
                config.positional_encoding,
                patch_count,
                config.model_dimension,
            ),
            requires_grad=config.learn_positional_encoding,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder = TSTEncoder(config)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        channels = patches.shape[1]
        projected = self.patch_projection(patches.permute(0, 1, 3, 2))
        encoded = projected.reshape(
            projected.shape[0] * channels,
            projected.shape[2],
            projected.shape[3],
        )
        encoded = self.embedding_dropout(encoded + self.position_embedding)
        encoded = self.encoder(encoded)
        encoded = encoded.reshape(
            -1, channels, encoded.shape[-2], encoded.shape[-1]
        )
        return encoded.permute(0, 1, 3, 2)


class ReferenceAlignedPatchTSTBackbone(nn.Module):
    channels = 3

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_count = (
            (config.history_points - config.patch_length) // config.patch_stride + 1
        )
        if config.patch_padding_end:
            self.patch_count += 1
            self.padding = nn.ReplicationPad1d((0, config.patch_stride))
        else:
            self.padding = None
        self.revin = (
            ReversibleInstanceNorm(
                self.channels,
                affine=config.revin_affine,
                subtract_last=config.revin_subtract_last,
            )
            if config.revin_enabled
            else None
        )
        self.encoder = ChannelIndependentEncoder(config, self.patch_count)
        self.head = FlattenHead(
            config, config.model_dimension * self.patch_count
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        revin_state = None
        if self.revin is not None:
            values, revin_state = self.revin.normalize(values.transpose(1, 2))
            values = values.transpose(1, 2)
        if self.padding is not None:
            values = self.padding(values)
        patches = values.unfold(
            dimension=-1,
            size=self.config.patch_length,
            step=self.config.patch_stride,
        ).permute(0, 1, 3, 2)
        encoded = self.encoder(patches)
        prediction = self.head(encoded)
        if self.revin is not None and revin_state is not None:
            prediction = self.revin.denormalize(
                prediction.transpose(1, 2), revin_state
            ).transpose(1, 2)
        return prediction


def _position_encoding(kind: str, length: int, dimension: int) -> torch.Tensor:
    if kind.lower() == "zeros":
        encoding = torch.empty(length, dimension)
        nn.init.uniform_(encoding, -0.02, 0.02)
        return encoding
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * -(math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    encoding = encoding - encoding.mean()
    return encoding / (encoding.std() * 10.0)
