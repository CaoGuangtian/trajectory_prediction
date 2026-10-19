"""CLI for the position-hold trajectory-prediction baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.baselines.position_hold import (
    PositionHoldSource,
    evaluate_position_hold,
    load_position_hold_config,
)
from trajectory_prediction.datasets.opensky import OpenSkySubset, discover_subset_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the position-hold baseline on standardized OpenSky data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/position_hold_v1.yaml"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--subset",
        type=int,
        action="append",
        dest="subsets",
        help="Subset number to evaluate; repeat for multiple subsets.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every discovered subset; all must have standardized data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing result for the same selection and version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_position_hold_config(args.config)
    dataset_root = config.dataset_root.resolve()
    discovered = discover_subset_ids(dataset_root)
    if not discovered:
        raise SystemExit(f"No complete OpenSky subsets found under {dataset_root}")
    if args.all:
        subset_ids = discovered
        selection_name = "all_subsets"
    elif args.subsets:
        subset_ids = sorted(set(args.subsets))
        selection_name = _selection_name(subset_ids)
    else:
        subset_ids = list(config.subsets)
        selection_name = _selection_name(subset_ids)
    missing_raw = sorted(set(subset_ids) - set(discovered))
    if missing_raw:
        raise SystemExit(f"Requested raw subsets are unavailable: {missing_raw}")

    sources: list[PositionHoldSource] = []
    missing_standardized: list[int] = []
    for subset_id in subset_ids:
        subset = OpenSkySubset(dataset_root, subset_id)
        if not subset.output_path.is_file():
            missing_standardized.append(subset_id)
            continue
        sources.append(
            PositionHoldSource(
                subset_id=subset_id,
                parquet_path=subset.output_path,
                manifest_path=subset.manifest_path,
            )
        )
    if missing_standardized:
        raise SystemExit(
            "Standardized data is missing for subsets: "
            f"{missing_standardized}. Run preprocessing first."
        )

    output_directory = (
        config.output_root.resolve() / selection_name / config.version
    )
    print(
        f"evaluating position hold on subsets {subset_ids} using split={config.split}",
        flush=True,
    )
    manifest = evaluate_position_hold(
        sources,
        config,
        output_directory,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def _selection_name(subset_ids: list[int]) -> str:
    if len(subset_ids) == 1:
        return f"subset_{subset_ids[0]}"
    return "subsets_" + "_".join(str(value) for value in subset_ids)


if __name__ == "__main__":
    main()
