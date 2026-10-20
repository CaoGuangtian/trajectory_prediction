"""Train the IMM-PatchTST residual fusion model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trajectory_prediction.models.imm_patchtst import load_imm_patchtst_config
from trajectory_prediction.models.imm_patchtst.cache import cache_directory
from trajectory_prediction.models.imm_patchtst.training import train_imm_patchtst

from .train_patchtst import _resolve_sources, _selection_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMM-PatchTST residual fusion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/imm_patchtst_residual_v1.yaml"),
    )
    parser.add_argument("--subset", type=int, action="append", dest="subsets")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imm_patchtst_config(args.config)
    patch_config = config.patchtst
    if args.device is not None:
        patch_config = replace(patch_config, device=args.device)
    if args.cuda_device is not None:
        patch_config = replace(
            patch_config, cuda_device_index=args.cuda_device
        )
    config = replace(config, patchtst=patch_config)
    subset_ids, sources = _resolve_sources(patch_config, args.subsets)
    selection = _selection_name(subset_ids)
    output_directory = (
        config.output_root.resolve() / selection / config.version / "training"
    )
    manifest = train_imm_patchtst(
        sources,
        config,
        cache_directory(config, selection, "train"),
        cache_directory(config, selection, "validation"),
        output_directory,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
