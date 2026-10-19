"""CLI for causal IMM-CV-CT-Singer trajectory prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.datasets.opensky import OpenSkySubset, discover_subset_ids
from trajectory_prediction.models.imm_kalman import (
    ImmKalmanSource,
    evaluate_imm_kalman,
    load_imm_kalman_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate causal open-loop IMM-CV-CT-Singer prediction on "
            "standardized OpenSky trajectories."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/imm_cv_ct_singer_v1.yaml"),
    )
    parser.add_argument(
        "--subset",
        type=int,
        action="append",
        dest="subsets",
        help="Subset number to evaluate; repeat for multiple subsets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing result for the same selection and version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imm_kalman_config(args.config)
    dataset_root = config.dataset_root.resolve()
    discovered = discover_subset_ids(dataset_root)
    if not discovered:
        raise SystemExit(f"No complete OpenSky subsets found under {dataset_root}")
    subset_ids = (
        sorted(set(args.subsets)) if args.subsets else list(config.subsets)
    )
    missing_raw = sorted(set(subset_ids) - set(discovered))
    if missing_raw:
        raise SystemExit(f"Requested raw subsets are unavailable: {missing_raw}")

    sources: list[ImmKalmanSource] = []
    missing_standardized: list[int] = []
    for subset_id in subset_ids:
        subset = OpenSkySubset(dataset_root, subset_id)
        if not subset.output_path.is_file():
            missing_standardized.append(subset_id)
            continue
        sources.append(
            ImmKalmanSource(
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

    selection_name = _selection_name(subset_ids)
    output_directory = (
        config.output_root.resolve() / selection_name / config.version
    )
    print(
        "evaluating IMM-CV-CT-Singer on subsets "
        f"{subset_ids} using split={config.split}",
        flush=True,
    )
    manifest = evaluate_imm_kalman(
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
