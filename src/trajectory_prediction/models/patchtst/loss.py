"""Masked objectives with an optional PyTorch dependency."""

from __future__ import annotations

from typing import Any

from .runtime import require_torch


def masked_huber_components(
    prediction: Any,
    target: Any,
    target_mask: Any,
    delta: float,
    torch_module: Any | None = None,
) -> tuple[Any, Any]:
    """Return the summed Huber loss and number of valid scalar coordinates."""

    torch = torch_module or require_torch()
    mask = target_mask.unsqueeze(-1).expand_as(prediction)
    difference = torch.abs(prediction - target)
    quadratic = torch.minimum(
        difference,
        torch.tensor(delta, device=difference.device, dtype=difference.dtype),
    )
    linear = difference - quadratic
    element_loss = 0.5 * quadratic.square() + delta * linear
    return element_loss.masked_select(mask).sum(), mask.sum()

