"""Runtime selection shared by PatchTST training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RuntimeDevice:
    device: Any
    requested: str
    resolved: str
    mixed_precision: bool


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PatchTST requires PyTorch. Install a CPU or CUDA build as described in "
            "docs/patchtst.md; the base preprocessing environment intentionally does "
            "not pin a hardware-specific PyTorch package."
        ) from error
    return torch


def resolve_runtime_device(
    requested: str,
    cuda_device_index: int,
    use_mixed_precision: bool,
    torch_module: Any | None = None,
) -> RuntimeDevice:
    torch = torch_module or require_torch()
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot access a CUDA GPU. "
            "Use --device cpu/auto or install a matching CUDA build of PyTorch."
        )
    if requested == "cuda" or (requested == "auto" and cuda_available):
        count = int(torch.cuda.device_count())
        if cuda_device_index >= count:
            raise RuntimeError(
                f"CUDA device {cuda_device_index} was requested, but only {count} "
                "device(s) are visible."
            )
        resolved = f"cuda:{cuda_device_index}"
    else:
        resolved = "cpu"
    return RuntimeDevice(
        device=torch.device(resolved),
        requested=requested,
        resolved=resolved,
        mixed_precision=bool(use_mixed_precision and resolved.startswith("cuda")),
    )


def configure_runtime(
    random_seed: int,
    cpu_threads: int,
    runtime: RuntimeDevice,
    torch_module: Any | None = None,
) -> None:
    torch = torch_module or require_torch()
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_num_threads(cpu_threads)
    if runtime.resolved.startswith("cuda"):
        torch.cuda.set_device(runtime.device)
        torch.cuda.manual_seed_all(random_seed)

