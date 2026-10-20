"""Persistent causal Standard-IMM predictions aligned to PatchTST windows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trajectory_prediction.models.patchtst.data import (
    PatchTSTSource,
    TrajectoryWindowCorpus,
)
from trajectory_prediction.models.standard_imm.core import (
    predict_standard_imm_batch,
)

from .config import ImmPatchTSTConfig


POSITIONS_FILE = "positions_enu.npy"
PROBABILITIES_FILE = "issue_model_probabilities.npy"
FALLBACK_FILE = "fallback_mask.npy"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True, slots=True)
class ImmPredictionCache:
    directory: Path
    positions_enu: np.ndarray
    issue_model_probabilities: np.ndarray
    fallback_mask: np.ndarray
    metadata: dict[str, Any]


def cache_directory(
    config: ImmPatchTSTConfig,
    selection: str,
    split: str,
) -> Path:
    return config.cache_root.resolve() / selection / config.version / split


def build_imm_prediction_cache(
    sources: Iterable[PatchTSTSource],
    split: str,
    config: ImmPatchTSTConfig,
    directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate all 1..prediction_points IMM forecasts without future access."""

    selected_sources = tuple(sorted(sources, key=lambda source: source.subset_id))
    corpus = _build_corpus(selected_sources, split, config)
    if not len(corpus):
        raise RuntimeError(f"IMM-PatchTST cache has no eligible {split} windows.")
    destination = directory.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"IMM-PatchTST cache already exists; use --overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    horizons = tuple(range(1, config.patchtst.prediction_points + 1))
    model_count = len(config.imm.model_names)
    try:
        positions = np.lib.format.open_memmap(
            temporary / POSITIONS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(corpus), len(horizons), 3),
        )
        probabilities = np.lib.format.open_memmap(
            temporary / PROBABILITIES_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(corpus), model_count),
        )
        fallback = np.lib.format.open_memmap(
            temporary / FALLBACK_FILE,
            mode="w+",
            dtype=np.bool_,
            shape=(len(corpus),),
        )
        for start in range(0, len(corpus), config.cache_batch_size):
            stop = min(start + config.cache_batch_size, len(corpus))
            indices = np.arange(start, stop, dtype=np.int64)
            batch = corpus.batch(
                indices,
                scales=None,
                apply_input_transform=False,
            )
            observation_mask = np.ones(
                batch.history_observation_mask.shape, dtype=bool
            )
            if config.imm.skip_interpolated_history_updates:
                observation_mask &= ~batch.history_is_interpolated
            if config.imm.skip_suspicious_history_updates:
                observation_mask &= ~batch.history_is_suspicious
            prediction = predict_standard_imm_batch(
                batch.history_enu,
                observation_mask,
                horizons,
                config.imm,
            )
            if not np.all(np.isfinite(prediction.positions_enu)):
                raise RuntimeError("Standard IMM cache prediction contains non-finite values.")
            if not np.all(np.isfinite(prediction.issue_model_probabilities)):
                raise RuntimeError("Standard IMM cache probabilities are non-finite.")
            if not np.allclose(
                prediction.issue_model_probabilities.sum(axis=1), 1.0, atol=1.0e-6
            ):
                raise RuntimeError("Standard IMM cache probabilities do not sum to one.")
            positions[start:stop] = prediction.positions_enu.astype(
                np.float32, copy=False
            )
            probabilities[start:stop] = (
                prediction.issue_model_probabilities.astype(np.float32, copy=False)
            )
            fallback[start:stop] = prediction.fallback_mask
        positions.flush()
        probabilities.flush()
        fallback.flush()
        del positions, probabilities, fallback

        metadata = _cache_metadata(corpus, split, config, horizons)
        (temporary / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def open_imm_prediction_cache(
    directory: Path,
    corpus: TrajectoryWindowCorpus,
    split: str,
    config: ImmPatchTSTConfig,
) -> ImmPredictionCache:
    resolved = directory.resolve()
    metadata_path = resolved / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"IMM-PatchTST cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = _cache_metadata(
        corpus,
        split,
        config,
        tuple(range(1, config.patchtst.prediction_points + 1)),
    )
    checked_keys = (
        "format_version",
        "fusion_config_hash",
        "patchtst_config_hash",
        "imm_config_hash",
        "split",
        "window_count",
        "prediction_points",
        "corpus_fingerprint",
    )
    mismatches = [
        key for key in checked_keys if metadata.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            "IMM-PatchTST cache does not match the selected corpus/configuration: "
            + ", ".join(mismatches)
        )
    positions = np.load(resolved / POSITIONS_FILE, mmap_mode="r")
    probabilities = np.load(resolved / PROBABILITIES_FILE, mmap_mode="r")
    fallback = np.load(resolved / FALLBACK_FILE, mmap_mode="r")
    expected_shapes = {
        POSITIONS_FILE: (len(corpus), config.patchtst.prediction_points, 3),
        PROBABILITIES_FILE: (len(corpus), len(config.imm.model_names)),
        FALLBACK_FILE: (len(corpus),),
    }
    actual_shapes = {
        POSITIONS_FILE: positions.shape,
        PROBABILITIES_FILE: probabilities.shape,
        FALLBACK_FILE: fallback.shape,
    }
    invalid = [
        name for name, shape in expected_shapes.items() if actual_shapes[name] != shape
    ]
    if invalid:
        raise ValueError(
            "IMM-PatchTST cache arrays have invalid shapes: " + ", ".join(invalid)
        )
    return ImmPredictionCache(
        directory=resolved,
        positions_enu=positions,
        issue_model_probabilities=probabilities,
        fallback_mask=fallback,
        metadata=metadata,
    )


def build_corpus_for_split(
    sources: Iterable[PatchTSTSource],
    split: str,
    config: ImmPatchTSTConfig,
) -> TrajectoryWindowCorpus:
    return _build_corpus(
        tuple(sorted(sources, key=lambda source: source.subset_id)), split, config
    )


def corpus_fingerprint(corpus: TrajectoryWindowCorpus) -> str:
    hasher = hashlib.sha256()
    hasher.update(corpus.split.encode("utf-8"))
    hasher.update(
        np.asarray(corpus.origin_segment_indices, dtype="<i8").tobytes()
    )
    hasher.update(np.asarray(corpus.origin_local_indices, dtype="<i8").tobytes())
    for segment in corpus.segments:
        identity = (
            segment.subset_id,
            segment.aircraft,
            segment.trajectory_id,
            segment.segment_id,
            segment.split,
            int(segment.times.size),
            float(segment.times[0]),
            float(segment.times[-1]),
        )
        hasher.update(
            json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    return hasher.hexdigest()


def _build_corpus(
    sources: tuple[PatchTSTSource, ...],
    split: str,
    config: ImmPatchTSTConfig,
) -> TrajectoryWindowCorpus:
    stride_by_split = {
        "train": config.patchtst.train_origin_stride_seconds,
        "validation": config.patchtst.validation_origin_stride_seconds,
        "test": config.patchtst.test_origin_stride_seconds,
    }
    try:
        stride = stride_by_split[split]
    except KeyError as error:
        raise ValueError("split must be train, validation, or test.") from error
    return TrajectoryWindowCorpus(sources, split, config.patchtst, stride)


def _cache_metadata(
    corpus: TrajectoryWindowCorpus,
    split: str,
    config: ImmPatchTSTConfig,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "artifact": "imm_patchtst_prediction_cache",
        "fusion_config_hash": config.config_hash,
        "patchtst_config_hash": config.patchtst.config_hash,
        "imm_config_hash": config.imm.config_hash,
        "split": split,
        "window_count": len(corpus),
        "prediction_points": config.patchtst.prediction_points,
        "horizons_seconds": list(horizons),
        "model_names": list(config.imm.model_names),
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "coordinate_system": "forecast_origin_local_enu_metres",
        "observation_policy": {
            "skip_interpolated_history_updates": (
                config.imm.skip_interpolated_history_updates
            ),
            "skip_suspicious_history_updates": (
                config.imm.skip_suspicious_history_updates
            ),
        },
    }
