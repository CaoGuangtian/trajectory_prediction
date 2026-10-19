"""Fixed-parameter standard IMM filter with causal open-loop forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_prediction.models.imm_kalman.core import (
    POSITION_INDICES,
    STATE_DIM,
    _fill_ct_model,
    _fill_cv_model,
    _fill_singer_model,
    _imm_forecast_means,
    _imm_predict,
    _initial_covariance,
    _measurement_update,
)

from .config import StandardImmConfig


@dataclass(frozen=True, slots=True)
class StandardImmPredictionBatch:
    positions_enu: np.ndarray
    fallback_mask: np.ndarray
    issue_model_probabilities: np.ndarray


def predict_standard_imm_batch(
    history_enu: np.ndarray,
    observation_mask: np.ndarray,
    horizons_seconds: tuple[int, ...],
    config: StandardImmConfig,
) -> StandardImmPredictionBatch:
    histories = np.asarray(history_enu, dtype=float)
    masks = np.asarray(observation_mask, dtype=bool)
    if histories.ndim != 3 or histories.shape[2] != 3:
        raise ValueError("history_enu must have shape (batch, history, 3).")
    if masks.shape != histories.shape[:2]:
        raise ValueError("observation_mask must match the history dimensions.")
    if histories.shape[1] != config.history_seconds:
        raise ValueError(f"Expected {config.history_seconds} history points, got {histories.shape[1]}.")
    if not horizons_seconds:
        raise ValueError("At least one prediction horizon is required.")

    batch_size, history_points, _ = histories.shape
    model_count = len(config.model_names)
    counts = masks.sum(axis=1)
    first = np.argmax(masks, axis=1)
    last = history_points - 1 - np.argmax(masks[:, ::-1], axis=1)
    spans = np.where(counts > 0, last - first, 0)
    fallback = (counts < config.minimum_history_points) | (spans < config.minimum_history_span_seconds)

    dynamics, process = _fixed_model_matrices(batch_size, config)
    transition = np.broadcast_to(
        np.asarray(config.model_transition_matrix, dtype=float),
        (batch_size, model_count, model_count),
    ).copy()
    probabilities = np.broadcast_to(
        np.asarray(config.initial_model_probabilities, dtype=float),
        (batch_size, model_count),
    ).copy()
    states = np.zeros((batch_size, model_count, STATE_DIM), dtype=float)
    states[:, :, POSITION_INDICES] = histories[:, None, 0, :]
    covariance = _initial_covariance(batch_size, config.initial_covariance_scale, model_count)
    measurement_covariance = np.diag(config.measurement_variances)

    states, covariance, probabilities = _measurement_update(
        states, covariance, probabilities, histories[:, 0, :], masks[:, 0], measurement_covariance
    )
    for history_index in range(1, history_points):
        states, covariance, probabilities = _imm_predict(
            states, covariance, probabilities, transition, dynamics, process
        )
        states, covariance, probabilities = _measurement_update(
            states,
            covariance,
            probabilities,
            histories[:, history_index, :],
            masks[:, history_index],
            measurement_covariance,
        )

    # Histories are expressed relative to the issue point, whose observed
    # position is exactly zero. Anchoring removes harmless filter position lag
    # without changing the filtered velocity, acceleration, or model weights.
    states[:, :, POSITION_INDICES] = 0.0
    issue_probabilities = probabilities.copy()
    selected = set(int(value) for value in horizons_seconds)
    predictions: dict[int, np.ndarray] = {}
    for step in range(1, max(selected) + 1):
        states, probabilities = _imm_forecast_means(states, probabilities, transition, dynamics)
        if step in selected:
            combined = np.sum(probabilities[:, :, None] * states, axis=1)
            predictions[step] = combined[:, POSITION_INDICES]
    positions = np.stack([predictions[int(value)] for value in horizons_seconds], axis=1)

    invalid = ~np.all(np.isfinite(positions), axis=(1, 2))
    fallback |= invalid
    if np.any(fallback):
        positions[fallback] = 0.0
        issue_probabilities[fallback] = np.asarray(config.initial_model_probabilities)
    return StandardImmPredictionBatch(positions, fallback, issue_probabilities)


def _fixed_model_matrices(
    batch_size: int,
    config: StandardImmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    model_count = len(config.model_names)
    dynamics = np.zeros((batch_size, model_count, STATE_DIM, STATE_DIM), dtype=float)
    process = np.zeros_like(dynamics)
    cv_q = np.broadcast_to(np.asarray(config.cv_process_q_xyz), (batch_size, 3))
    ct_q = np.broadcast_to(np.asarray(config.ct_process_q_xyz), (batch_size, 3))
    singer_alpha = np.broadcast_to(np.asarray(config.singer_alpha_xyz), (batch_size, 3))
    singer_sigma = np.broadcast_to(np.asarray(config.singer_sigma_xyz), (batch_size, 3))
    for index, name in enumerate(config.model_names):
        if name == "cv":
            _fill_cv_model(dynamics[:, index], process[:, index], cv_q)
        elif name in {"ct_left", "ct_right"}:
            sign = 1.0 if name == "ct_left" else -1.0
            turn_rate = np.full(batch_size, sign * config.ct_turn_rate_rad_s)
            _fill_ct_model(dynamics[:, index], process[:, index], turn_rate, ct_q)
        elif name == "singer":
            _fill_singer_model(dynamics[:, index], process[:, index], singer_alpha, singer_sigma)
        else:  # Defensive; configuration validation rejects this first.
            raise ValueError(f"Unsupported motion model: {name}")
    process = 0.5 * (process + np.swapaxes(process, -1, -2))
    diagonal = np.arange(STATE_DIM)
    process[:, :, diagonal, diagonal] += 1.0e-9
    return dynamics, process
