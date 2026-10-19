"""CLI for training PatchTST on train/validation only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trajectory_prediction.datasets.opensky import OpenSkySubset, discover_subset_ids
from trajectory_prediction.models.patchtst import PatchTSTSource, load_patchtst_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trajectory PatchTST.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/models/patchtst_raw_v1.yaml")
    )
    parser.add_argument("--subset", type=int, action="append", dest="subsets")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    output_directory = (
        config.output_root.resolve() / selection / config.version / "training"
    )
    try:
        from trajectory_prediction.models.patchtst.training import train_patchtst

        manifest = train_patchtst(
            sources,
            config,
            output_directory,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def _resolve_sources(
    config: object, requested_subsets: list[int] | None
) -> tuple[list[int], list[PatchTSTSource]]:
    dataset_root = config.dataset_root.resolve()
    discovered = discover_subset_ids(dataset_root)
    subset_ids = sorted(set(requested_subsets)) if requested_subsets else list(config.subsets)
    missing = sorted(set(subset_ids) - set(discovered))
    if missing:
        raise SystemExit(f"Requested raw subsets are unavailable: {missing}")
    sources: list[PatchTSTSource] = []
    for subset_id in subset_ids:
        subset = OpenSkySubset(dataset_root, subset_id)
        if not subset.output_path.is_file():
            raise SystemExit(f"Standardized subset is missing: {subset_id}")
        sources.append(
            PatchTSTSource(subset_id, subset.output_path, subset.manifest_path)
        )
    return subset_ids, sources


def _selection_name(subset_ids: list[int]) -> str:
    return (
        f"subset_{subset_ids[0]}"
        if len(subset_ids) == 1
        else "subsets_" + "_".join(str(value) for value in subset_ids)
    )


if __name__ == "__main__":
    main()

