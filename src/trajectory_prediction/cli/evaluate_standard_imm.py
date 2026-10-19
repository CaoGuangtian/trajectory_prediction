"""CLI for fixed-parameter standard IMM and CV-KF evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.datasets.opensky import OpenSkySubset, discover_subset_ids
from trajectory_prediction.models.standard_imm import (
    StandardImmSource,
    evaluate_standard_imm,
    load_standard_imm_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Standard IMM or CV-KF.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/standard_imm_v1.yaml"),
    )
    parser.add_argument("--subset", type=int, action="append", dest="subsets")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_standard_imm_config(args.config)
    dataset_root = config.dataset_root.resolve()
    discovered = discover_subset_ids(dataset_root)
    subset_ids = sorted(set(args.subsets)) if args.subsets else list(config.subsets)
    missing = sorted(set(subset_ids) - set(discovered))
    if missing:
        raise SystemExit(f"Requested raw subsets are unavailable: {missing}")
    sources: list[StandardImmSource] = []
    for subset_id in subset_ids:
        subset = OpenSkySubset(dataset_root, subset_id)
        if not subset.output_path.is_file():
            raise SystemExit(f"Standardized subset is missing: {subset_id}")
        sources.append(StandardImmSource(subset_id, subset.output_path, subset.manifest_path))
    selection = (
        f"subset_{subset_ids[0]}"
        if len(subset_ids) == 1
        else "subsets_" + "_".join(str(value) for value in subset_ids)
    )
    output_directory = config.output_root.resolve() / selection / config.version
    manifest = evaluate_standard_imm(sources, config, output_directory, overwrite=args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
