"""Build causal Standard-IMM forecasts aligned to PatchTST windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.models.imm_patchtst import load_imm_patchtst_config
from trajectory_prediction.models.imm_patchtst.cache import (
    build_imm_prediction_cache,
    cache_directory,
)

from .train_patchtst import _resolve_sources, _selection_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IMM-PatchTST prediction caches.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/imm_patchtst_residual_v1.yaml"),
    )
    parser.add_argument("--subset", type=int, action="append", dest="subsets")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        action="append",
        dest="splits",
        help="Repeat to build multiple splits; defaults to train and validation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imm_patchtst_config(args.config)
    subset_ids, sources = _resolve_sources(config.patchtst, args.subsets)
    selection = _selection_name(subset_ids)
    splits = tuple(dict.fromkeys(args.splits or ("train", "validation")))
    manifests: list[dict[str, object]] = []
    for split in splits:
        directory = cache_directory(config, selection, split)
        metadata = build_imm_prediction_cache(
            sources,
            split,
            config,
            directory,
            overwrite=args.overwrite,
        )
        manifests.append({"directory": directory.as_posix(), **metadata})
    print(json.dumps(manifests, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
