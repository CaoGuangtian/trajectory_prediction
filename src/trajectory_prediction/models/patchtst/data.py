"""Causal fixed-window data preparation for trajectory PatchTST."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import PatchTSTConfig


EARTH_MEAN_RADIUS_M = 6_371_008.8
SOURCE_COLUMNS = (
    "aircraft",
    "trajectory_id",
    "segment_id",
    "point_index",
    "timeAtServer",
    "longitude",
    "latitude",
    "geoAltitude",
    "trusted",
    "split",
    "is_interpolated",
    "is_speed_suspicious",
)


@dataclass(frozen=True, slots=True)
class PatchTSTSource:
    subset_id: int
    parquet_path: Path
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class NormalizationScales:
    east_m: float
    north_m: float
    up_m: float
    fitted_split: str
    sampled_origins: int

    @property
    def array(self) -> np.ndarray:
        return np.asarray([self.east_m, self.north_m, self.up_m], dtype=np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "root_mean_square_about_forecast_origin",
            "east_m": self.east_m,
            "north_m": self.north_m,
            "up_m": self.up_m,
            "fitted_split": self.fitted_split,
            "sampled_origins": self.sampled_origins,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySegment:
    subset_id: int
    aircraft: int
    trajectory_id: int
    segment_id: int
    trusted: str
    split: str
    point_index: np.ndarray
    times: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    geo_altitude: np.ndarray
    is_interpolated: np.ndarray
    is_suspicious: np.ndarray


@dataclass(frozen=True, slots=True)
class WindowBatch:
    history_enu: np.ndarray
    future_enu: np.ndarray
    target_mask: np.ndarray
    segment_indices: np.ndarray
    origin_indices: np.ndarray
    origin_longitude: np.ndarray
    origin_latitude: np.ndarray
    origin_altitude: np.ndarray
    target_longitude: np.ndarray
    target_latitude: np.ndarray
    target_altitude: np.ndarray


class TrajectoryWindowCorpus:
    """In-memory segment index whose windows never cross a segment boundary."""

    def __init__(
        self,
        sources: Iterable[PatchTSTSource],
        split: str,
        config: PatchTSTConfig,
        origin_stride_seconds: int,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test.")
        if origin_stride_seconds < 1:
            raise ValueError("origin_stride_seconds must be positive.")
        self.split = split
        self.config = config
        self.origin_stride_seconds = origin_stride_seconds
        self.sources = tuple(sorted(sources, key=lambda source: source.subset_id))
        if not self.sources:
            raise ValueError("At least one standardized source is required.")
        subset_ids = [source.subset_id for source in self.sources]
        if len(subset_ids) != len(set(subset_ids)):
            raise ValueError("PatchTST sources contain duplicate subset IDs.")

        segments: list[TrajectorySegment] = []
        origin_segments: list[np.ndarray] = []
        origin_indices: list[np.ndarray] = []
        for source in self.sources:
            source_segments = _read_segments(source, split, config.trusted_values)
            for segment in source_segments:
                eligible = _eligible_origins(segment, config, origin_stride_seconds)
                if eligible.size:
                    origin_segments.append(
                        np.full(eligible.size, len(segments), dtype=np.int32)
                    )
                    origin_indices.append(eligible.astype(np.int32, copy=False))
                segments.append(segment)
        self.segments = tuple(segments)
        self.origin_segment_indices = (
            np.concatenate(origin_segments)
            if origin_segments
            else np.empty(0, dtype=np.int32)
        )
        self.origin_local_indices = (
            np.concatenate(origin_indices)
            if origin_indices
            else np.empty(0, dtype=np.int32)
        )

    def __len__(self) -> int:
        return int(self.origin_local_indices.size)

    def batch(
        self,
        indices: np.ndarray | list[int],
        scales: NormalizationScales | None = None,
    ) -> WindowBatch:
        item_indices = np.asarray(indices, dtype=np.int64)
        if item_indices.ndim != 1:
            raise ValueError("Window indices must be one-dimensional.")
        if item_indices.size and (
            item_indices.min() < 0 or item_indices.max() >= len(self)
        ):
            raise IndexError("Window index is outside the corpus.")
        batch_size = item_indices.size
        history_points = self.config.history_points
        prediction_points = self.config.prediction_points
        history = np.empty((batch_size, history_points, 3), dtype=np.float32)
        future = np.empty((batch_size, prediction_points, 3), dtype=np.float32)
        target_mask = np.empty((batch_size, prediction_points), dtype=bool)
        origin_longitude = np.empty(batch_size, dtype=np.float64)
        origin_latitude = np.empty(batch_size, dtype=np.float64)
        origin_altitude = np.empty(batch_size, dtype=np.float64)
        target_longitude = np.empty((batch_size, prediction_points), dtype=np.float64)
        target_latitude = np.empty((batch_size, prediction_points), dtype=np.float64)
        target_altitude = np.empty((batch_size, prediction_points), dtype=np.float64)
        segment_indices = self.origin_segment_indices[item_indices]
        origins = self.origin_local_indices[item_indices]

        for segment_index in np.unique(segment_indices):
            rows = np.flatnonzero(segment_indices == segment_index)
            segment = self.segments[int(segment_index)]
            group_origins = origins[rows].astype(np.int64)
            history_offsets = np.arange(-history_points + 1, 1, dtype=np.int64)
            future_offsets = np.arange(1, prediction_points + 1, dtype=np.int64)
            history_indices = group_origins[:, None] + history_offsets[None, :]
            future_indices = group_origins[:, None] + future_offsets[None, :]

            lon0 = segment.longitude[group_origins]
            lat0 = segment.latitude[group_origins]
            alt0 = segment.geo_altitude[group_origins]
            cosine_latitude = np.maximum(np.cos(np.radians(lat0)), 1.0e-6)

            history[rows, :, 0] = (
                np.radians(segment.longitude[history_indices] - lon0[:, None])
                * EARTH_MEAN_RADIUS_M
                * cosine_latitude[:, None]
            )
            history[rows, :, 1] = (
                np.radians(segment.latitude[history_indices] - lat0[:, None])
                * EARTH_MEAN_RADIUS_M
            )
            history[rows, :, 2] = segment.geo_altitude[history_indices] - alt0[:, None]
            future[rows, :, 0] = (
                np.radians(segment.longitude[future_indices] - lon0[:, None])
                * EARTH_MEAN_RADIUS_M
                * cosine_latitude[:, None]
            )
            future[rows, :, 1] = (
                np.radians(segment.latitude[future_indices] - lat0[:, None])
                * EARTH_MEAN_RADIUS_M
            )
            future[rows, :, 2] = segment.geo_altitude[future_indices] - alt0[:, None]

            valid = np.ones(future_indices.shape, dtype=bool)
            if self.config.exclude_interpolated_targets:
                valid &= ~segment.is_interpolated[future_indices]
            if self.config.exclude_suspicious_targets:
                valid &= ~segment.is_suspicious[future_indices]
            target_mask[rows] = valid
            origin_longitude[rows] = lon0
            origin_latitude[rows] = lat0
            origin_altitude[rows] = alt0
            target_longitude[rows] = segment.longitude[future_indices]
            target_latitude[rows] = segment.latitude[future_indices]
            target_altitude[rows] = segment.geo_altitude[future_indices]

        if scales is not None:
            scale_array = scales.array[None, None, :]
            history /= scale_array
            future /= scale_array
        return WindowBatch(
            history_enu=history,
            future_enu=future,
            target_mask=target_mask,
            segment_indices=segment_indices.copy(),
            origin_indices=origins.copy(),
            origin_longitude=origin_longitude,
            origin_latitude=origin_latitude,
            origin_altitude=origin_altitude,
            target_longitude=target_longitude,
            target_latitude=target_latitude,
            target_altitude=target_altitude,
        )


def compute_normalization_scales(
    corpus: TrajectoryWindowCorpus,
    maximum_origins: int,
    minimum_scale_m: float,
    batch_size: int = 2048,
) -> NormalizationScales:
    """Fit axis scales from train windows only, using deterministic sampling."""

    if corpus.split != "train":
        raise ValueError("Normalization scales may only be fitted on the train split.")
    if not len(corpus):
        raise ValueError("The train corpus has no eligible windows.")
    sample_count = min(len(corpus), int(maximum_origins))
    indices = np.linspace(0, len(corpus) - 1, sample_count, dtype=np.int64)
    sum_squares = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for start in range(0, sample_count, batch_size):
        batch = corpus.batch(indices[start : start + batch_size])
        history = batch.history_enu.astype(np.float64)
        sum_squares += np.square(history).sum(axis=(0, 1))
        counts += history.shape[0] * history.shape[1]
        valid_future = batch.target_mask[:, :, None]
        sum_squares += np.where(valid_future, np.square(batch.future_enu), 0.0).sum(
            axis=(0, 1)
        )
        counts += valid_future.sum(axis=(0, 1))
    scales = np.sqrt(sum_squares / np.maximum(counts, 1))
    scales = np.maximum(scales, float(minimum_scale_m))
    return NormalizationScales(
        east_m=float(scales[0]),
        north_m=float(scales[1]),
        up_m=float(scales[2]),
        fitted_split="train",
        sampled_origins=sample_count,
    )


def save_normalization_scales(path: Path, scales: NormalizationScales) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(scales.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_normalization_scales(path: Path) -> NormalizationScales:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scales = NormalizationScales(
        east_m=float(payload["east_m"]),
        north_m=float(payload["north_m"]),
        up_m=float(payload["up_m"]),
        fitted_split=str(payload["fitted_split"]),
        sampled_origins=int(payload["sampled_origins"]),
    )
    if scales.fitted_split != "train" or np.any(scales.array <= 0):
        raise ValueError("Invalid PatchTST normalization artifact.")
    return scales


def _read_segments(
    source: PatchTSTSource,
    split: str,
    trusted_values: tuple[str, ...],
) -> list[TrajectorySegment]:
    if not source.parquet_path.is_file():
        raise FileNotFoundError(
            f"Standardized trajectory artifact does not exist: {source.parquet_path}"
        )
    table = pq.read_table(
        source.parquet_path,
        columns=list(SOURCE_COLUMNS),
        filters=[("split", "=", split)],
    )
    if trusted_values:
        table = table.filter(
            pc.is_in(table["trusted"], value_set=pa.array(trusted_values))
        )
    if table.num_rows == 0:
        return []
    data = {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in SOURCE_COLUMNS
    }
    boundaries = np.flatnonzero(
        (data["aircraft"][1:] != data["aircraft"][:-1])
        | (data["trajectory_id"][1:] != data["trajectory_id"][:-1])
        | (data["segment_id"][1:] != data["segment_id"][:-1])
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [table.num_rows]))
    segments: list[TrajectorySegment] = []
    for start, end in zip(starts, ends, strict=True):
        point_index = np.asarray(data["point_index"][start:end], dtype=np.int64)
        times = np.asarray(data["timeAtServer"][start:end], dtype=np.float64)
        if point_index.size > 1 and (
            np.any(np.diff(point_index) != 1)
            or np.any(np.abs(np.diff(times) - 1.0) > 1.0e-6)
        ):
            raise ValueError(
                "Standardized segment is not contiguous at one hertz: "
                f"subset={source.subset_id}, segment={int(data['segment_id'][start])}"
            )
        segments.append(
            TrajectorySegment(
                subset_id=source.subset_id,
                aircraft=int(data["aircraft"][start]),
                trajectory_id=int(data["trajectory_id"][start]),
                segment_id=int(data["segment_id"][start]),
                trusted=str(data["trusted"][start]),
                split=split,
                point_index=point_index,
                times=times,
                longitude=np.asarray(data["longitude"][start:end], dtype=np.float64),
                latitude=np.asarray(data["latitude"][start:end], dtype=np.float64),
                geo_altitude=np.asarray(data["geoAltitude"][start:end], dtype=np.float64),
                is_interpolated=np.asarray(
                    data["is_interpolated"][start:end], dtype=bool
                ),
                is_suspicious=np.asarray(
                    data["is_speed_suspicious"][start:end], dtype=bool
                ),
            )
        )
    return segments


def _eligible_origins(
    segment: TrajectorySegment,
    config: PatchTSTConfig,
    stride: int,
) -> np.ndarray:
    first = config.history_points - 1
    last = segment.times.size - config.prediction_points - 1
    if last < first:
        return np.empty(0, dtype=np.int32)
    origins = np.arange(first, last + 1, dtype=np.int64)
    offsets = np.rint(segment.times[origins] - segment.times[0]).astype(np.int64)
    mask = offsets % stride == 0
    if config.exclude_interpolated_origins:
        mask &= ~segment.is_interpolated[origins]
    if config.exclude_suspicious_origins:
        mask &= ~segment.is_suspicious[origins]
    return origins[mask].astype(np.int32, copy=False)
