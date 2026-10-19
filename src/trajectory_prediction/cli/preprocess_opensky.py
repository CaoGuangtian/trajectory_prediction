"""CLI for producing compact OpenSky trajectory files for model input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.datasets.opensky import (
    OpenSkySubset,
    discover_subset_ids,
    load_dataset_aircraft_ids,
)
from trajectory_prediction.preprocess.config import load_config
from trajectory_prediction.preprocess.pipeline import preprocess_subset
from trajectory_prediction.preprocess.split import (
    build_aircraft_split,
    write_split_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OpenSky subsets into compact per-trajectory Parquet files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/opensky_standardized_1hz_v2.yaml"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--subset",
        type=int,
        action="append",
        dest="subsets",
        help="Subset number to process; repeat for multiple subsets.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process every discovered subset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing derived artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = config.dataset_root.resolve()
    discovered = discover_subset_ids(dataset_root)
    if not discovered:
        raise SystemExit(f"No complete OpenSky subsets found under {dataset_root}")
    if args.all:
        subset_ids = discovered
    elif args.subsets:
        subset_ids = sorted(set(args.subsets))
    else:
        subset_ids = list(config.subsets)
    missing = sorted(set(subset_ids) - set(discovered))
    if missing:
        raise SystemExit(f"Requested subsets are unavailable: {missing}")

    aircraft_ids = load_dataset_aircraft_ids(dataset_root)
    split_map = build_aircraft_split(aircraft_ids, config)
    split_path, _ = write_split_artifact(
        dataset_root / "manifests",
        split_map,
        config,
        overwrite=args.overwrite,
    )
    print(f"aircraft split: {split_path} ({len(split_map):,} aircraft)", flush=True)

    summaries: list[dict[str, object]] = []
    for subset_id in subset_ids:
        subset = OpenSkySubset(dataset_root, subset_id)
        print(f"processing subset_{subset_id}: {subset.track_path}", flush=True)
        manifest = preprocess_subset(
            subset,
            config,
            split_map,
            overwrite=args.overwrite,
        )
        stats = manifest["statistics"]
        assert isinstance(stats, dict)
        print(
            f"wrote {subset.output_path}: rows={stats['output_rows']:,}, "
            f"aircraft={stats['output_aircraft']:,}, "
            f"segments={stats['segment_count']:,}",
            flush=True,
        )
        summaries.append(
            {
                "subset": subset_id,
                "output": subset.output_path.as_posix(),
                "statistics": stats,
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
