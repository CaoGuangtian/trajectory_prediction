"""OpenSky/LocaRDS dataset discovery and metadata loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_TRACK_COLUMNS = (
    "id",
    "timeAtServer",
    "aircraft",
    "latitude",
    "longitude",
    "baroAltitude",
    "geoAltitude",
    "numMeasurements",
    "measurements",
)


@dataclass(frozen=True, slots=True)
class OpenSkySubset:
    """Resolved input and output paths for one OpenSky subset."""

    dataset_root: Path
    subset_id: int

    @property
    def directory(self) -> Path:
        return self.dataset_root / f"subset_{self.subset_id}"

    @property
    def track_path(self) -> Path:
        return self.directory / f"set_{self.subset_id}.parquet"

    @property
    def aircraft_path(self) -> Path:
        return self.directory / f"set_{self.subset_id}_aircraft.csv"

    @property
    def output_directory(self) -> Path:
        return self.directory / "standardized_1hz"

    @property
    def output_path(self) -> Path:
        return self.output_directory / f"set_{self.subset_id}_standardized_1hz.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "manifest.yaml"

    def validate(self) -> None:
        if not self.track_path.is_file():
            raise FileNotFoundError(f"OpenSky track file does not exist: {self.track_path}")
        if not self.aircraft_path.is_file():
            raise FileNotFoundError(
                f"OpenSky aircraft metadata does not exist: {self.aircraft_path}"
            )
        columns = tuple(pq.read_schema(self.track_path).names)
        if columns != EXPECTED_TRACK_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {self.track_path}: "
                f"expected {EXPECTED_TRACK_COLUMNS}, got {columns}"
            )


def discover_subset_ids(dataset_root: Path) -> list[int]:
    """Return subset numbers that contain both required input files."""

    result: list[int] = []
    if not dataset_root.is_dir():
        return result
    for directory in dataset_root.glob("subset_*"):
        if not directory.is_dir():
            continue
        suffix = directory.name.removeprefix("subset_")
        if not suffix.isdigit():
            continue
        subset_id = int(suffix)
        subset = OpenSkySubset(dataset_root, subset_id)
        if subset.track_path.is_file() and subset.aircraft_path.is_file():
            result.append(subset_id)
    return sorted(result)


def load_aircraft_metadata(path: Path) -> dict[int, str]:
    """Load the per-aircraft trusted flag as true/false/unknown."""

    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["aircraft", "trusted"]:
            raise ValueError(
                f"Unexpected aircraft metadata header in {path}: {reader.fieldnames}"
            )
        for row in reader:
            aircraft = int(row["aircraft"])
            raw = row["trusted"].strip().upper()
            trusted = "true" if raw == "TRUE" else "false" if raw == "FALSE" else "unknown"
            result[aircraft] = trusted
    return result


def load_dataset_aircraft_ids(dataset_root: Path) -> list[int]:
    """Return the union of aircraft identifiers in every discovered subset."""

    aircraft_ids: set[int] = set()
    for subset_id in discover_subset_ids(dataset_root):
        subset = OpenSkySubset(dataset_root, subset_id)
        aircraft_ids.update(load_aircraft_metadata(subset.aircraft_path))
    return sorted(aircraft_ids)
