"""CLI for held-out PatchTST evaluation and frontend-compatible artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trajectory_prediction.models.patchtst import load_patchtst_config
from trajectory_prediction.models.patchtst.evaluation import evaluate_patchtst

from .train_patchtst import _resolve_sources, _selection_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained trajectory PatchTST.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/models/patchtst_raw_v1.yaml")
    )
    parser.add_argument("--subset", type=int, action="append", dest="subsets")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_patchtst_config(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.cuda_device is not None:
        config = replace(config, cuda_device_index=args.cuda_device)
    subset_ids, sources = _resolve_sources(config, args.subsets)
    selection = _selection_name(subset_ids)
    output_directory = config.output_root.resolve() / selection / config.version
    training_directory = output_directory / "training"
    checkpoint_path = (args.checkpoint or training_directory / "checkpoints/best.pt").resolve()
    normalization_path = (
        args.normalization or training_directory / "normalization.json"
    ).resolve()
    try:
        manifest = evaluate_patchtst(
            sources,
            config,
            checkpoint_path,
            normalization_path,
            output_directory,
            overwrite=args.overwrite,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
