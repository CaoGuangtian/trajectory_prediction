#!/usr/bin/env python3
"""Stream an OpenSky CSV file into an equivalent Parquet file.

The large source CSV is read in record batches, so conversion does not require
loading the complete dataset into memory.  The ``measurements`` column remains
a string; this utility changes the storage format, not the data model.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/openskyDataset/subset_1/set_1.csv")
EXPECTED_COLUMNS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OpenSky LocaRDS CSV file to Parquet without loading "
            "the whole CSV into memory."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: input path with a .parquet suffix).",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "gzip", "brotli", "lz4", "none"),
        default="zstd",
        help="Parquet compression codec (default: zstd).",
    )
    parser.add_argument(
        "--block-size-mib",
        type=int,
        default=64,
        help="CSV streaming read block size in MiB (default: 64).",
    )
    parser.add_argument(
        "--row-group-rows",
        type=int,
        default=250_000,
        help="Maximum rows per Parquet row group (default: 250000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file after a successful conversion.",
    )
    return parser.parse_args()


def load_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. Install it in the traj-pre environment, "
            "then run this script again."
        ) from exc
    return pa, pacsv, pq


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_suffix(".parquet")
    )

    if not input_path.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise SystemExit(f"Input must be a .csv file: {input_path}")
    if output_path.suffix.lower() != ".parquet":
        raise SystemExit(f"Output must be a .parquet file: {output_path}")
    if input_path == output_path:
        raise SystemExit("Input and output paths must be different.")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it after a successful conversion."
        )
    if args.block_size_mib < 1:
        raise SystemExit("--block-size-mib must be at least 1.")
    if args.row_group_rows < 1:
        raise SystemExit("--row-group-rows must be at least 1.")

    return input_path, output_path


def opensky_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("timeAtServer", pa.float64()),
            pa.field("aircraft", pa.int64()),
            pa.field("latitude", pa.float64()),
            pa.field("longitude", pa.float64()),
            pa.field("baroAltitude", pa.float64()),
            pa.field("geoAltitude", pa.float64()),
            pa.field("numMeasurements", pa.int64()),
            pa.field("measurements", pa.string()),
        ]
    )


def convert_csv_to_parquet(args: argparse.Namespace) -> tuple[Path, int, int]:
    input_path, output_path = validate_args(args)
    pa, pacsv, pq = load_pyarrow()
    schema = opensky_schema(pa)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    row_count = 0
    batch_count = 0
    next_progress_row = 1_000_000
    writer = None

    try:
        reader = pacsv.open_csv(
            input_path,
            read_options=pacsv.ReadOptions(
                block_size=args.block_size_mib * 1024 * 1024,
                use_threads=True,
                encoding="utf8",
            ),
            parse_options=pacsv.ParseOptions(
                delimiter=",",
                quote_char='"',
                double_quote=True,
                newlines_in_values=False,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types=schema,
                null_values=["", "NA", "NaN", "null", "None"],
                strings_can_be_null=False,
                check_utf8=True,
            ),
        )

        actual_columns = tuple(reader.schema.names)
        if actual_columns != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected CSV columns. "
                f"Expected {EXPECTED_COLUMNS}, got {actual_columns}."
            )

        compression = None if args.compression == "none" else args.compression
        writer = pq.ParquetWriter(
            temporary_path,
            schema=reader.schema,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )

        for batch in reader:
            writer.write_batch(batch, row_group_size=args.row_group_rows)
            row_count += batch.num_rows
            batch_count += 1
            if row_count >= next_progress_row:
                print(f"converted {row_count:,} rows...", flush=True)
                next_progress_row = ((row_count // 1_000_000) + 1) * 1_000_000

        writer.close()
        writer = None

        metadata = pq.read_metadata(temporary_path)
        if metadata.num_rows != row_count:
            raise RuntimeError(
                "Parquet row-count verification failed: "
                f"wrote {row_count}, metadata reports {metadata.num_rows}."
            )
        if tuple(pq.read_schema(temporary_path).names) != EXPECTED_COLUMNS:
            raise RuntimeError("Parquet column verification failed.")

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output appeared during conversion and was not replaced: {output_path}"
            )
        os.replace(temporary_path, output_path)
        return output_path, row_count, batch_count
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    try:
        output_path, row_count, batch_count = convert_csv_to_parquet(args)
    except KeyboardInterrupt:
        raise SystemExit("Conversion cancelled; incomplete temporary output removed.")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Conversion failed: {exc}") from exc

    output_size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"wrote: {output_path}")
    print(f"rows: {row_count:,}")
    print(f"record batches: {batch_count:,}")
    print(f"output size: {output_size_mib:,.2f} MiB")


if __name__ == "__main__":
    main()
