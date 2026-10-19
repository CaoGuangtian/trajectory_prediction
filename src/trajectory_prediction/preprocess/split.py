"""Stable aircraft-level train/validation/test assignment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import PreprocessConfig


SPLIT_SCHEMA = pa.schema(
    [
        pa.field("aircraft", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


def build_aircraft_split(
    aircraft_ids: list[int], config: PreprocessConfig
) -> dict[int, str]:
    ranked = sorted(
        aircraft_ids,
        key=lambda aircraft: hashlib.sha256(
            f"{config.split.seed}:{aircraft}".encode("ascii")
        ).digest(),
    )
    count = len(ranked)
    train_end = int(count * config.split.train_ratio)
    validation_end = train_end + int(count * config.split.validation_ratio)
    result: dict[int, str] = {}
    for index, aircraft in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        result[aircraft] = split
    return result


def write_split_artifact(
    directory: Path,
    split_map: dict[int, str],
    config: PreprocessConfig,
    overwrite: bool,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / "aircraft_split.parquet"
    manifest_path = directory / "aircraft_split_manifest.yaml"
    if parquet_path.exists() and not overwrite:
        table = pq.read_table(parquet_path)
        existing = dict(zip(table["aircraft"].to_pylist(), table["split"].to_pylist()))
        if existing != split_map:
            raise FileExistsError(
                f"Existing split artifact differs from the requested split: {parquet_path}"
            )
        return parquet_path, manifest_path

    rows = [
        {"aircraft": aircraft, "split": split_map[aircraft]}
        for aircraft in sorted(split_map)
    ]
    table = pa.Table.from_pylist(rows, schema=SPLIT_SCHEMA)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{parquet_path.name}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        pq.write_table(table, temporary_path, compression=config.parquet_compression)
        os.replace(temporary_path, parquet_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    counts = {name: list(split_map.values()).count(name) for name in ("train", "validation", "test")}
    manifest = {
        "artifact": "opensky_aircraft_split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "seed": config.split.seed,
        "aircraft_count": len(split_map),
        "counts": counts,
        "path": parquet_path.as_posix(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return parquet_path, manifest_path
