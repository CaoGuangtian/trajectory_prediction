"""End-to-end OpenSky trajectory preprocessing with no persistent intermediates."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from trajectory_prediction.datasets.opensky import (
    OpenSkySubset,
    load_aircraft_metadata,
)

from .config import PreprocessConfig
from .deduplicate import DeduplicatedPoint, collect_source_stats, iter_deduplicated_points
from .resample import resample_aircraft
from .segment import SegmentStats, split_runs_on_speed


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("aircraft", pa.int32(), nullable=False),
        pa.field("trajectory_id", pa.int64(), nullable=False),
        pa.field("segment_id", pa.int64(), nullable=False),
        pa.field("point_index", pa.int32(), nullable=False),
        pa.field("timeAtServer", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("geoAltitude", pa.float64(), nullable=False),
        pa.field("baroAltitude", pa.float64()),
        pa.field("trusted", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("is_interpolated", pa.bool_(), nullable=False),
        pa.field("is_speed_suspicious", pa.bool_(), nullable=False),
        pa.field("source_record_count", pa.int32(), nullable=False),
        pa.field("numMeasurements", pa.int32(), nullable=False),
    ]
)


def _aircraft_groups(
    points: Iterable[DeduplicatedPoint],
) -> Iterable[tuple[int, list[DeduplicatedPoint]]]:
    current_aircraft: int | None = None
    current: list[DeduplicatedPoint] = []
    for point in points:
        if current_aircraft is None:
            current_aircraft = point.aircraft
        if point.aircraft != current_aircraft:
            yield current_aircraft, current
            current_aircraft = point.aircraft
            current = []
        current.append(point)
    if current_aircraft is not None:
        yield current_aircraft, current


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def preprocess_subset(
    subset: OpenSkySubset,
    config: PreprocessConfig,
    split_map: dict[int, str],
    overwrite: bool = False,
) -> dict[str, object]:
    subset.validate()
    if subset.output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {subset.output_path}. Use --overwrite to replace it."
        )
    subset.output_directory.mkdir(parents=True, exist_ok=True)
    metadata = load_aircraft_metadata(subset.aircraft_path)
    missing_split = sorted(set(metadata) - set(split_map))
    if missing_split:
        raise ValueError(f"Aircraft are missing from the global split: {missing_split[:10]}")

    duckdb_temporary = tempfile.TemporaryDirectory(
        prefix=".duckdb_sort_",
        dir=subset.output_directory,
    )
    connection = duckdb.connect()
    temp_directory = duckdb_temporary.name.replace("'", "''")
    memory_limit = config.duckdb_memory_limit.replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_directory}'")
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(f"SET threads = {config.duckdb_threads}")
    connection.execute("SET preserve_insertion_order = false")
    source_stats = collect_source_stats(
        connection,
        subset.track_path,
        config.coordinate_conflict_threshold_m,
    )

    temporary_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{subset.output_path.name}.",
        suffix=".tmp",
        dir=subset.output_directory,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, object]] = []
    output_rows = 0
    interpolated_rows = 0
    output_aircraft: set[int] = set()
    trajectory_count = 0
    segment_count = 0
    segment_id = 0
    trajectory_id = 0
    segment_stats_total = Counter()
    split_counts = Counter()
    trusted_counts = Counter()

    def flush() -> None:
        nonlocal buffer
        if not buffer or writer is None:
            return
        table = pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA)
        writer.write_table(table, row_group_size=len(buffer))
        buffer = []

    try:
        writer = pq.ParquetWriter(
            temporary_path,
            OUTPUT_SCHEMA,
            compression=config.parquet_compression,
            use_dictionary=["trusted", "split"],
            write_statistics=True,
        )
        iterator = iter_deduplicated_points(
            connection,
            subset.track_path,
            config.coordinate_conflict_threshold_m,
        )
        for aircraft, raw_points in _aircraft_groups(iterator):
            runs = resample_aircraft(
                raw_points,
                config.sampling_interval_seconds,
                config.macro_gap_seconds,
                config.max_interpolation_missing_steps,
            )
            segments, segment_stats = split_runs_on_speed(
                runs,
                config.sampling_interval_seconds,
                config.suspicious_horizontal_speed_mps,
                config.strong_anomaly_horizontal_speed_mps,
                config.minimum_segment_points,
                config.minimum_segment_duration_seconds,
            )
            for key, value in asdict(segment_stats).items():
                segment_stats_total[key] += value
            if not segments:
                continue

            trajectory_lookup: dict[int, int] = {}
            trusted = metadata.get(aircraft, "unknown")
            split = split_map[aircraft]
            for segment in segments:
                if segment.trajectory_index not in trajectory_lookup:
                    trajectory_id += 1
                    trajectory_count += 1
                    trajectory_lookup[segment.trajectory_index] = trajectory_id
                segment_id += 1
                segment_count += 1
                rows: list[dict[str, object]] = []
                for point_index, point in enumerate(segment.points):
                    rows.append(
                        {
                            "aircraft": aircraft,
                            "trajectory_id": trajectory_lookup[segment.trajectory_index],
                            "segment_id": segment_id,
                            "point_index": point_index,
                            "timeAtServer": point.timestamp,
                            "longitude": point.longitude,
                            "latitude": point.latitude,
                            "geoAltitude": point.geo_altitude,
                            "baroAltitude": point.baro_altitude,
                            "trusted": trusted,
                            "split": split,
                            "is_interpolated": point.is_interpolated,
                            "is_speed_suspicious": point.is_speed_suspicious,
                            "source_record_count": point.source_record_count,
                            "numMeasurements": point.num_measurements,
                        }
                    )
                if buffer and len(buffer) + len(rows) > config.parquet_row_group_rows:
                    flush()
                buffer.extend(rows)
                output_rows += len(rows)
                interpolated_rows += sum(row["is_interpolated"] for row in rows)
                output_aircraft.add(aircraft)
                split_counts[split] += len(rows)
                trusted_counts[trusted] += len(rows)
        flush()
        writer.close()
        writer = None
        connection.close()
        duckdb_temporary.cleanup()

        metadata_check = pq.read_metadata(temporary_path)
        if metadata_check.num_rows != output_rows:
            raise RuntimeError(
                f"Output row verification failed: expected {output_rows}, "
                f"Parquet reports {metadata_check.num_rows}."
            )
        if tuple(pq.read_schema(temporary_path).names) != tuple(OUTPUT_SCHEMA.names):
            raise RuntimeError("Output schema verification failed.")
        os.replace(temporary_path, subset.output_path)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        connection.close()
        duckdb_temporary.cleanup()
        temporary_path.unlink(missing_ok=True)
        raise

    output_size = subset.output_path.stat().st_size
    manifest: dict[str, object] = {
        "artifact": "opensky_standardized_trajectory_points_1hz",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": subset.subset_id,
        "source": {
            "tracks": subset.track_path.as_posix(),
            "aircraft_metadata": subset.aircraft_path.as_posix(),
        },
        "output": {
            "path": subset.output_path.as_posix(),
            "size_bytes": output_size,
            "schema": OUTPUT_SCHEMA.names,
            "ordering": ["aircraft", "trajectory_id", "segment_id", "point_index"],
        },
        "configuration": config.as_manifest_dict(),
        "statistics": {
            **asdict(source_stats),
            "output_rows": output_rows,
            "output_aircraft": len(output_aircraft),
            "trajectory_count": trajectory_count,
            "segment_count": segment_count,
            "interpolated_rows": interpolated_rows,
            **dict(segment_stats_total),
            "rows_by_split": dict(sorted(split_counts.items())),
            "rows_by_trusted": dict(sorted(trusted_counts.items())),
        },
        "notes": [
            "The source Parquet was not modified.",
            "Rows without valid longitude, latitude, and geoAltitude are excluded.",
            "Each non-empty 1 s bin is represented by robust median observations.",
            "Altitude observations are not motion-model smoothed; smoothing belongs to each model input pipeline.",
            (
                "Empty bins are interpolated only when adjacent observed-bin timestamps "
                f"are at most {config.macro_gap_seconds:g} s apart; inserted rows are "
                "marked by is_interpolated=true."
            ),
            (
                "Final segments shorter than "
                f"{config.minimum_segment_duration_seconds:g} s are excluded from the output."
            ),
            "No model windows are materialized.",
            "JSON syntax is used because JSON is a valid YAML 1.2 subset.",
        ],
    }
    _write_manifest(subset.manifest_path, manifest)
    return manifest
