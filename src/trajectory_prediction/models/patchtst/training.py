"""Train PatchTST without touching the held-out test split."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import PatchTSTConfig
from .data import (
    NormalizationScales,
    PatchTSTSource,
    TrajectoryWindowCorpus,
    compute_normalization_scales,
    save_normalization_scales,
)
from .loss import masked_huber_components
from .runtime import configure_runtime, require_torch, resolve_runtime_device


class _CorpusIndexDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


class _CorpusCollator:
    def __init__(
        self,
        corpus: TrajectoryWindowCorpus,
        scales: NormalizationScales,
    ) -> None:
        self.corpus = corpus
        self.scales = scales

    def __call__(self, indices: list[int]) -> tuple[Any, Any, Any]:
        torch = require_torch()
        batch = self.corpus.batch(indices, self.scales)
        return (
            torch.from_numpy(batch.history_enu),
            torch.from_numpy(batch.future_enu),
            torch.from_numpy(batch.target_mask),
        )


def train_patchtst(
    sources: Iterable[PatchTSTSource],
    config: PatchTSTConfig,
    output_directory: Path,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Fit on train, select on validation, and never instantiate a test corpus."""

    torch = require_torch()
    from .model import TrajectoryPatchTST, model_parameter_count

    selected_sources = tuple(sorted(sources, key=lambda source: source.subset_id))
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_directory / "best.pt"
    last_path = checkpoint_directory / "last.pt"
    scales_path = output_directory / "normalization.json"
    history_path = output_directory / "training_history.csv"
    manifest_path = output_directory / "training_manifest.yaml"
    existing = [
        path
        for path in (best_path, last_path, scales_path, history_path, manifest_path)
        if path.exists()
    ]
    if existing and not overwrite and not resume:
        raise FileExistsError(
            "PatchTST training artifacts already exist; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    runtime = resolve_runtime_device(
        config.device,
        config.cuda_device_index,
        config.use_mixed_precision,
        torch,
    )
    configure_runtime(config.random_seed, config.cpu_threads, runtime, torch)
    training_started = time.perf_counter()
    preparation_started = time.perf_counter()
    train_corpus = TrajectoryWindowCorpus(
        selected_sources, "train", config, config.train_origin_stride_seconds
    )
    validation_corpus = TrajectoryWindowCorpus(
        selected_sources,
        "validation",
        config,
        config.validation_origin_stride_seconds,
    )
    if not len(train_corpus) or not len(validation_corpus):
        raise RuntimeError("PatchTST requires eligible train and validation windows.")
    scales = compute_normalization_scales(
        train_corpus,
        config.normalization_max_origins,
        config.normalization_min_scale_m,
    )
    save_normalization_scales(scales_path, scales)

    generator = torch.Generator().manual_seed(config.random_seed)
    loader_options = {
        "num_workers": config.dataloader_workers,
        "pin_memory": runtime.resolved.startswith("cuda"),
        "persistent_workers": config.dataloader_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        _CorpusIndexDataset(len(train_corpus)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_CorpusCollator(train_corpus, scales),
        **loader_options,
    )
    validation_loader = torch.utils.data.DataLoader(
        _CorpusIndexDataset(len(validation_corpus)),
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        collate_fn=_CorpusCollator(validation_corpus, scales),
        **loader_options,
    )
    model = TrajectoryPatchTST(config).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=runtime.mixed_precision)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=runtime.mixed_precision)
    preparation_seconds = time.perf_counter() - preparation_started
    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {last_path}")
        checkpoint = torch.load(last_path, map_location=runtime.device, weights_only=False)
        _validate_checkpoint(checkpoint, config)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, config.maximum_epochs + 1):
        epoch_started = time.perf_counter()
        train_phase_started = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            runtime,
            config.huber_delta,
            torch,
            optimizer=optimizer,
            scaler=scaler,
        )
        train_seconds = time.perf_counter() - train_phase_started
        validation_phase_started = time.perf_counter()
        validation_loss = _run_epoch(
            model,
            validation_loader,
            runtime,
            config.huber_delta,
            torch,
        )
        validation_seconds = time.perf_counter() - validation_phase_started
        epoch_seconds = time.perf_counter() - epoch_started
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_huber": train_loss,
            "validation_huber": validation_loss,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "is_best": improved,
        }
        history.append(row)
        checkpoint = {
            "format_version": 1,
            "config_hash": config.config_hash,
            "config": config.as_manifest_dict(),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "normalization": scales.as_dict(),
        }
        _save_checkpoint_atomic(last_path, checkpoint, torch)
        if improved:
            _save_checkpoint_atomic(best_path, checkpoint, torch)
        _write_history(history_path, history)
        print(
            f"epoch={epoch} train_huber={train_loss:.6f} "
            f"validation_huber={validation_loss:.6f} "
            f"train_seconds={train_seconds:.1f} "
            f"validation_seconds={validation_seconds:.1f} "
            f"epoch_seconds={epoch_seconds:.1f} device={runtime.resolved}",
            flush=True,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    manifest: dict[str, object] = {
        "artifact": "patchtst_training",
        "model_type": config.model_type,
        "display_name": config.display_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.as_manifest_dict(),
        "data_policy": {
            "training_split": "train",
            "model_selection_split": "validation",
            "test_split_accessed": False,
            "normalization_fitted_on": "train",
        },
        "runtime": {
            "requested_device": runtime.requested,
            "resolved_device": runtime.resolved,
            "mixed_precision": runtime.mixed_precision,
            "torch_version": torch.__version__,
        },
        "sources": [_source_entry(source) for source in selected_sources],
        "statistics": {
            "train_windows": len(train_corpus),
            "validation_windows": len(validation_corpus),
            "completed_epochs": len(history),
            "best_validation_huber": best_validation_loss,
            "model_parameters": model_parameter_count(model),
            "data_preparation_seconds": preparation_seconds,
            "training_wall_seconds": time.perf_counter() - training_started,
        },
        "outputs": {
            "best_checkpoint": best_path.as_posix(),
            "last_checkpoint": last_path.as_posix(),
            "normalization": scales_path.as_posix(),
            "training_history": history_path.as_posix(),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _run_epoch(
    model: Any,
    loader: Any,
    runtime: Any,
    huber_delta: float,
    torch: Any,
    optimizer: Any | None = None,
    scaler: Any | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for history, target, target_mask in loader:
            history = history.to(runtime.device, non_blocking=True)
            target = target.to(runtime.device, non_blocking=True)
            target_mask = target_mask.to(runtime.device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=runtime.mixed_precision,
            ):
                prediction = model(history)
                loss_sum, valid_count = masked_huber_components(
                    prediction, target, target_mask, huber_delta, torch
                )
                loss = loss_sum / valid_count.clamp_min(1)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss_sum.detach().cpu())
            total_count += int(valid_count.detach().cpu())
    if not total_count:
        raise RuntimeError("No valid target coordinates were available for loss computation.")
    return total_loss / total_count


def _validate_checkpoint(checkpoint: dict[str, object], config: PatchTSTConfig) -> None:
    if checkpoint.get("config_hash") != config.config_hash:
        raise ValueError(
            "Checkpoint configuration hash does not match the selected configuration."
        )


def _save_checkpoint_atomic(path: Path, payload: dict[str, object], torch: Any) -> None:
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_huber",
                    "validation_huber",
                    "train_seconds",
                    "validation_seconds",
                    "epoch_seconds",
                    "is_best",
                ],
            )
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _source_entry(source: PatchTSTSource) -> dict[str, object]:
    entry: dict[str, object] = {
        "subset": source.subset_id,
        "path": source.parquet_path.as_posix(),
    }
    if source.manifest_path is not None and source.manifest_path.is_file():
        payload = json.loads(source.manifest_path.read_text(encoding="utf-8"))
        entry.update(
            {
                "manifest": source.manifest_path.as_posix(),
                "preprocessing_version": payload.get("configuration", {}).get(
                    "version"
                ),
                "preprocessing_config_hash": payload.get("configuration", {}).get(
                    "config_hash"
                ),
            }
        )
    return entry
