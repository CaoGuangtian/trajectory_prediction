"""Vectorized causal IMM filter and open-loop trajectory predictor."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .config import ImmKalmanConfig


MODEL_NAMES = ("cv", "ct", "singer")
STATE_DIM = 9
POSITION_INDICES = np.array([0, 3, 6])
VELOCITY_INDICES = np.array([1, 4, 7])
ACCELERATION_INDICES = np.array([2, 5, 8])


@dataclass(frozen=True, slots=True)
class ImmBatchParameters:
    cv_q: np.ndarray
    ct_turn_rate: np.ndarray
    ct_q: np.ndarray
    singer_alpha: np.ndarray
    singer_sigma: np.ndarray
    transition_matrix: np.ndarray
    initial_probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class ImmPredictionBatch:
    positions_enu: np.ndarray
    fallback_mask: np.ndarray
    issue_model_probabilities: np.ndarray
    maneuver_mask: np.ndarray
    turn_maneuver_mask: np.ndarray
    acceleration_maneuver_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _MotionGate:
    cv_velocity: np.ndarray
    cv_valid: np.ndarray
    turn_mask: np.ndarray
    acceleration_mask: np.ndarray
    turn_rate: np.ndarray
    acceleration: np.ndarray

    @property
    def maneuver_mask(self) -> np.ndarray:
        return self.turn_mask | self.acceleration_mask


def predict_imm_batch(
    history_enu: np.ndarray,
    observation_mask: np.ndarray,
    horizons_seconds: tuple[int, ...],
    config: ImmKalmanConfig,
) -> ImmPredictionBatch:
    """Predict selected horizons from fixed-length, forecast-origin ENU histories."""

    histories = np.asarray(history_enu, dtype=float)
    masks = np.asarray(observation_mask, dtype=bool)
    if histories.ndim != 3 or histories.shape[2] != 3:
        raise ValueError("history_enu must have shape (batch, history, 3).")
    if masks.shape != histories.shape[:2]:
        raise ValueError("observation_mask must match the first two history dimensions.")
    if histories.shape[1] != config.history_seconds:
        raise ValueError(
            f"Expected {config.history_seconds} history points, got {histories.shape[1]}."
        )
    if not horizons_seconds:
        raise ValueError("At least one prediction horizon is required.")

    batch_size, history_points, _ = histories.shape
    motion_gate = _identify_motion_gate(histories, masks, config)
    parameters, fallback_mask, fitted_velocity, fitted_acceleration = (
        _identify_parameters(histories, masks, config)
    )
    history_dynamics, process_covariance = _model_matrices(parameters)

    states = np.zeros((batch_size, len(MODEL_NAMES), STATE_DIM), dtype=float)
    states[:, :, POSITION_INDICES] = histories[:, None, 0, :]
    covariance = _initial_covariance(
        batch_size,
        config.initial_covariance_scale,
    )
    probabilities = parameters.initial_probabilities.copy()
    measurement_covariance = np.diag(config.measurement_variances)

    states, covariance, probabilities = _measurement_update(
        states,
        covariance,
        probabilities,
        histories[:, 0, :],
        masks[:, 0],
        measurement_covariance,
    )
    for history_index in range(1, history_points):
        states, covariance, probabilities = _imm_predict(
            states,
            covariance,
            probabilities,
            parameters.transition_matrix,
            history_dynamics,
            process_covariance,
        )
        states, covariance, probabilities = _measurement_update(
            states,
            covariance,
            probabilities,
            histories[:, history_index, :],
            masks[:, history_index],
            measurement_covariance,
        )

    forecast_dynamics = history_dynamics
    forecast_model_transition = parameters.transition_matrix
    states[:, :, POSITION_INDICES] = 0.0
    if config.motion_gate_enabled:
        # The recent least-squares estimate is materially more robust to
        # one-hertz position noise than raw first/second differences.  It is
        # shared by all models so their forecast starts from the same causal
        # motion state.
        states[:, :, VELOCITY_INDICES] = motion_gate.cv_velocity[:, None, :]
        states[:, :, ACCELERATION_INDICES] = motion_gate.acceleration[:, None, :]
        forecast_parameters = replace(
            parameters,
            ct_turn_rate=motion_gate.turn_rate,
        )
        forecast_dynamics, _ = _model_matrices(forecast_parameters)
        allowed_models = np.column_stack(
            [
                np.ones(batch_size, dtype=bool),
                motion_gate.turn_mask,
                motion_gate.acceleration_mask,
            ]
        )
        probabilities *= allowed_models
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        forecast_model_transition = _gate_transition_matrix(
            parameters.transition_matrix,
            allowed_models,
        )
    issue_probabilities = probabilities.copy()
    selected = set(int(value) for value in horizons_seconds)
    maximum_horizon = max(selected)
    prediction_lookup: dict[int, np.ndarray] = {}
    for step in range(1, maximum_horizon + 1):
        states, probabilities = _imm_forecast_means(
            states,
            probabilities,
            forecast_model_transition,
            forecast_dynamics,
        )
        if step in selected:
            combined = np.sum(probabilities[:, :, None] * states, axis=1)
            prediction_lookup[step] = combined[:, POSITION_INDICES]

    predictions = np.stack(
        [prediction_lookup[int(value)] for value in horizons_seconds],
        axis=1,
    )
    horizon_values = np.asarray(horizons_seconds, dtype=float)
    cv_velocity = (
        np.where(
            motion_gate.cv_valid[:, None],
            motion_gate.cv_velocity,
            fitted_velocity,
        )
        if config.motion_gate_enabled
        else fitted_velocity
    )
    cv_predictions = cv_velocity[:, None, :] * horizon_values[None, :, None]
    use_cv = fallback_mask.copy()
    if config.motion_gate_enabled:
        use_cv |= ~motion_gate.maneuver_mask
    if np.any(use_cv):
        predictions[use_cv] = cv_predictions[use_cv]
        issue_probabilities[use_cv] = np.array([1.0, 0.0, 0.0])

    invalid = ~np.all(np.isfinite(predictions), axis=(1, 2))
    if np.any(invalid):
        predictions[invalid] = cv_predictions[invalid]
        fallback_mask = fallback_mask | invalid
        issue_probabilities[invalid] = np.array([1.0, 0.0, 0.0])

    return ImmPredictionBatch(
        positions_enu=predictions,
        fallback_mask=fallback_mask,
        issue_model_probabilities=issue_probabilities,
        maneuver_mask=(
            motion_gate.maneuver_mask
            if config.motion_gate_enabled
            else np.zeros(batch_size, dtype=bool)
        ),
        turn_maneuver_mask=(
            motion_gate.turn_mask
            if config.motion_gate_enabled
            else np.zeros(batch_size, dtype=bool)
        ),
        acceleration_maneuver_mask=(
            motion_gate.acceleration_mask
            if config.motion_gate_enabled
            else np.zeros(batch_size, dtype=bool)
        ),
    )


def _identify_motion_gate(
    histories: np.ndarray,
    masks: np.ndarray,
    config: ImmKalmanConfig,
) -> _MotionGate:
    """Identify persistent maneuvers from three causal overlapping blocks."""

    batch_size, history_points, _ = histories.shape
    recent_start = history_points - config.cv_fallback_history_points
    cv_velocity, cv_valid = _ols_velocity(
        histories,
        masks,
        recent_start,
        history_points,
        minimum_points=min(
            config.minimum_history_points,
            config.cv_fallback_history_points,
        ),
    )

    block_points = config.maneuver_gate_block_points
    required_points = 3 * block_points - 2
    block_start = history_points - required_points
    block_starts = (
        block_start,
        block_start + block_points - 1,
        block_start + 2 * (block_points - 1),
    )
    minimum_block_points = max(3, int(np.ceil(0.7 * block_points)))
    block_results = [
        _ols_velocity(
            histories,
            masks,
            start,
            start + block_points,
            minimum_points=minimum_block_points,
        )
        for start in block_starts
    ]
    velocities = np.stack([item[0] for item in block_results], axis=1)
    valid = np.column_stack([item[1] for item in block_results])
    horizontal = velocities[:, :, :2]
    speeds = np.linalg.norm(horizontal, axis=2)
    interval_s = float(block_points - 1)

    first_turn = _signed_angle(horizontal[:, 0], horizontal[:, 1]) / interval_s
    second_turn = _signed_angle(horizontal[:, 1], horizontal[:, 2]) / interval_s
    turn_consistency = _consistent_signed_change(
        first_turn,
        second_turn,
        config.maneuver_gate_min_turn_rate_rad_s,
        config.maneuver_gate_consistency_ratio,
    )
    turn_mask = (
        np.all(valid, axis=1)
        & np.all(speeds >= config.maneuver_gate_min_speed_m_s, axis=1)
        & turn_consistency
    )
    turn_rate = np.where(
        turn_mask,
        np.clip(
            0.5 * (first_turn + second_turn),
            -config.ct_turn_rate_max,
            config.ct_turn_rate_max,
        ),
        0.0,
    )

    first_acceleration = (speeds[:, 1] - speeds[:, 0]) / interval_s
    second_acceleration = (speeds[:, 2] - speeds[:, 1]) / interval_s
    acceleration_consistency = _consistent_signed_change(
        first_acceleration,
        second_acceleration,
        config.maneuver_gate_min_acceleration_m_s2,
        config.maneuver_gate_consistency_ratio,
    )
    acceleration_mask = (
        np.all(valid, axis=1)
        & (speeds[:, 2] >= config.maneuver_gate_min_speed_m_s)
        & acceleration_consistency
    )
    scalar_acceleration = np.where(
        acceleration_mask,
        0.5 * (first_acceleration + second_acceleration),
        0.0,
    )
    latest_direction = np.divide(
        horizontal[:, 2],
        speeds[:, 2, None],
        out=np.zeros((batch_size, 2), dtype=float),
        where=speeds[:, 2, None] > 1.0e-9,
    )
    acceleration = np.zeros((batch_size, 3), dtype=float)
    acceleration[:, :2] = scalar_acceleration[:, None] * latest_direction

    return _MotionGate(
        cv_velocity=cv_velocity,
        cv_valid=cv_valid,
        turn_mask=turn_mask,
        acceleration_mask=acceleration_mask,
        turn_rate=turn_rate,
        acceleration=acceleration,
    )


def _ols_velocity(
    histories: np.ndarray,
    masks: np.ndarray,
    start: int,
    stop: int,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = histories[:, start:stop, :]
    window_mask = masks[:, start:stop]
    weights = window_mask.astype(float)
    times = np.arange(start, stop, dtype=float) - (histories.shape[1] - 1)
    counts = weights.sum(axis=1)
    safe_counts = np.maximum(counts, 1.0)
    mean_time = np.sum(weights * times[None, :], axis=1) / safe_counts
    mean_position = (
        np.sum(weights[:, :, None] * positions, axis=1)
        / safe_counts[:, None]
    )
    centered_time = times[None, :] - mean_time[:, None]
    denominator = np.sum(weights * np.square(centered_time), axis=1)
    numerator = np.sum(
        weights[:, :, None]
        * centered_time[:, :, None]
        * (positions - mean_position[:, None, :]),
        axis=1,
    )
    velocity = np.divide(
        numerator,
        denominator[:, None],
        out=np.zeros_like(numerator),
        where=denominator[:, None] > 1.0e-9,
    )
    first = np.argmax(window_mask, axis=1)
    last = window_mask.shape[1] - 1 - np.argmax(window_mask[:, ::-1], axis=1)
    span = np.where(counts > 0, last - first, 0)
    valid = (
        (counts >= minimum_points)
        & (span >= max(2, (stop - start) // 2))
        & np.all(np.isfinite(velocity), axis=1)
    )
    velocity[~valid] = 0.0
    return velocity, valid


def _signed_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = np.sum(first * second, axis=1)
    return np.arctan2(cross, dot)


def _consistent_signed_change(
    first: np.ndarray,
    second: np.ndarray,
    minimum_magnitude: float,
    minimum_ratio: float,
) -> np.ndarray:
    first_abs = np.abs(first)
    second_abs = np.abs(second)
    maximum = np.maximum(first_abs, second_abs)
    ratio = np.divide(
        np.minimum(first_abs, second_abs),
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    return (
        (first * second > 0)
        & (first_abs >= minimum_magnitude)
        & (second_abs >= minimum_magnitude)
        & (ratio >= minimum_ratio)
    )


def _gate_transition_matrix(
    transition: np.ndarray,
    allowed_models: np.ndarray,
) -> np.ndarray:
    gated = transition * allowed_models[:, None, :]
    row_sums = gated.sum(axis=2, keepdims=True)
    return gated / np.maximum(row_sums, 1.0e-12)


def _identify_parameters(
    histories: np.ndarray,
    masks: np.ndarray,
    config: ImmKalmanConfig,
) -> tuple[ImmBatchParameters, np.ndarray, np.ndarray, np.ndarray]:
    batch_size, history_points, _ = histories.shape
    time_offsets = np.arange(history_points, dtype=float) - (history_points - 1)
    weights = masks.astype(float)
    counts = masks.sum(axis=1)
    first = np.argmax(masks, axis=1)
    last = history_points - 1 - np.argmax(masks[:, ::-1], axis=1)
    spans = np.where(counts > 0, last - first, 0)
    fallback = (
        (counts < config.minimum_history_points)
        | (spans < config.minimum_history_span_seconds)
    )

    x1 = time_offsets
    x2 = 0.5 * np.square(time_offsets)
    s11 = np.sum(weights * np.square(x1), axis=1)
    s12 = np.sum(weights * x1 * x2, axis=1)
    s22 = np.sum(weights * np.square(x2), axis=1)
    b1 = np.einsum("bt,t,btd->bd", weights, x1, histories)
    b2 = np.einsum("bt,t,btd->bd", weights, x2, histories)
    determinant = s11 * s22 - np.square(s12)
    safe_determinant = np.where(np.abs(determinant) > 1.0e-9, determinant, 1.0)
    velocity = (
        s22[:, None] * b1 - s12[:, None] * b2
    ) / safe_determinant[:, None]
    acceleration = (
        -s12[:, None] * b1 + s11[:, None] * b2
    ) / safe_determinant[:, None]
    singular = np.abs(determinant) <= 1.0e-9
    slope_denominator = np.where(s11 > 1.0e-9, s11, 1.0)
    simple_velocity = b1 / slope_denominator[:, None]
    velocity[singular] = simple_velocity[singular]
    acceleration[singular] = 0.0

    step_velocity = np.diff(histories, axis=1)
    velocity_mask = masks[:, 1:] & masks[:, :-1]
    step_acceleration = np.diff(step_velocity, axis=1)
    acceleration_mask = velocity_mask[:, 1:] & velocity_mask[:, :-1]

    q_axis = _masked_median(
        np.square(step_acceleration),
        acceleration_mask[:, :, None],
        axis=1,
        default=np.array([4.0, 4.0, 1.0]),
    )
    q_axis = np.clip(q_axis, config.process_q_min, config.process_q_max)
    cv_q = q_axis.copy()
    horizontal_q = np.maximum(q_axis[:, 0], q_axis[:, 1])
    ct_q = np.column_stack(
        [
            np.clip(1.5 * horizontal_q, config.process_q_min, config.process_q_max),
            np.clip(1.5 * horizontal_q, config.process_q_min, config.process_q_max),
            np.clip(q_axis[:, 2], config.process_q_min, config.process_q_max),
        ]
    )

    first_velocity = step_velocity[:, :-1, :2]
    second_velocity = step_velocity[:, 1:, :2]
    first_speed = np.linalg.norm(first_velocity, axis=2)
    second_speed = np.linalg.norm(second_velocity, axis=2)
    cross = (
        first_velocity[:, :, 0] * second_velocity[:, :, 1]
        - first_velocity[:, :, 1] * second_velocity[:, :, 0]
    )
    dot = np.sum(first_velocity * second_velocity, axis=2)
    turn_rates = np.arctan2(cross, dot)
    turn_mask = acceleration_mask & (first_speed > 5.0) & (second_speed > 5.0)
    ct_turn_rate = _masked_median(
        turn_rates,
        turn_mask,
        axis=1,
        default=np.zeros(batch_size),
    )
    ct_turn_rate = np.clip(
        ct_turn_rate,
        -config.ct_turn_rate_max,
        config.ct_turn_rate_max,
    )

    singer_alpha, singer_sigma = _estimate_singer(
        step_acceleration,
        acceleration_mask,
        config,
    )
    horizontal_acceleration = np.linalg.norm(step_acceleration[:, :, :2], axis=2)
    vertical_acceleration = np.abs(step_acceleration[:, :, 2])
    median_abs_turn = _masked_median(
        np.abs(turn_rates),
        turn_mask,
        axis=1,
        default=np.zeros(batch_size),
    )
    median_horizontal_acceleration = _masked_median(
        horizontal_acceleration,
        acceleration_mask,
        axis=1,
        default=np.zeros(batch_size),
    )
    median_vertical_acceleration = _masked_median(
        vertical_acceleration,
        acceleration_mask,
        axis=1,
        default=np.zeros(batch_size),
    )
    cv_score = 1.5 / (
        1.0
        + 18.0 * median_abs_turn
        + 0.08 * median_horizontal_acceleration
        + 0.05 * median_vertical_acceleration
    )
    ct_score = (
        0.5
        + 20.0 * np.abs(ct_turn_rate)
        + 0.03 * median_horizontal_acceleration
    )
    singer_score = (
        0.8
        + 0.05 * median_horizontal_acceleration
        + 0.08 * median_vertical_acceleration
        + 0.25 * np.mean(singer_sigma, axis=1)
    )
    scores = np.clip(
        np.column_stack([cv_score, ct_score, singer_score]),
        1.0e-6,
        None,
    )
    initial_probabilities = scores / scores.sum(axis=1, keepdims=True)
    stickiness = np.clip(
        0.90 + 0.06 * np.max(initial_probabilities, axis=1),
        config.model_stickiness_min,
        config.model_stickiness_max,
    )
    transition_matrix = np.zeros((batch_size, 3, 3), dtype=float)
    for source_model in range(3):
        transition_matrix[:, source_model, source_model] = stickiness
        targets = initial_probabilities.copy()
        targets[:, source_model] = 0.0
        targets /= targets.sum(axis=1, keepdims=True)
        transition_matrix[:, source_model, :] += (
            (1.0 - stickiness)[:, None] * targets
        )

    parameters = ImmBatchParameters(
        cv_q=cv_q,
        ct_turn_rate=ct_turn_rate,
        ct_q=ct_q,
        singer_alpha=singer_alpha,
        singer_sigma=singer_sigma,
        transition_matrix=transition_matrix,
        initial_probabilities=initial_probabilities,
    )
    return parameters, fallback, velocity, acceleration


def _estimate_singer(
    acceleration: np.ndarray,
    acceleration_mask: np.ndarray,
    config: ImmKalmanConfig,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = acceleration.shape[0]
    if acceleration.shape[1] < 2:
        return (
            np.full((batch_size, 3), 0.1),
            np.full((batch_size, 3), 1.0),
        )
    a0 = acceleration[:, :-1, :]
    a1 = acceleration[:, 1:, :]
    mask = acceleration_mask[:, :-1] & acceleration_mask[:, 1:]
    weights = mask[:, :, None].astype(float)
    counts = weights.sum(axis=1)
    safe_counts = np.where(counts > 0, counts, 1.0)
    mean0 = np.sum(weights * a0, axis=1) / safe_counts
    mean1 = np.sum(weights * a1, axis=1) / safe_counts
    centered0 = a0 - mean0[:, None, :]
    centered1 = a1 - mean1[:, None, :]
    variance0 = np.sum(weights * np.square(centered0), axis=1)
    covariance = np.sum(weights * centered0 * centered1, axis=1)
    beta = np.divide(
        covariance,
        variance0,
        out=np.zeros_like(covariance),
        where=variance0 > 1.0e-9,
    )
    alpha = 1.0 - beta
    alpha = np.where(
        (counts >= 2) & np.isfinite(alpha) & (alpha > 0),
        alpha,
        0.1,
    )
    alpha = np.clip(alpha, config.alpha_min, config.alpha_max)
    bias = mean1 - beta * mean0
    residual = a1 - (beta[:, None, :] * a0 + bias[:, None, :])
    residual_variance = np.sum(weights * np.square(residual), axis=1) / safe_counts
    sigma = np.sqrt(np.maximum(residual_variance, 0.0) / np.maximum(2.0 * alpha, 1.0e-9))
    sigma = np.where(
        (counts >= 2) & np.isfinite(sigma) & (sigma > 0),
        sigma,
        1.0,
    )
    sigma = np.clip(sigma, config.sigma_min, config.sigma_max)
    return alpha, sigma


def _masked_median(
    values: np.ndarray,
    mask: np.ndarray,
    axis: int,
    default: np.ndarray,
) -> np.ndarray:
    masked = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(masked, axis=axis)
    return np.where(np.isfinite(result), result, default)


def _model_matrices(
    parameters: ImmBatchParameters,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = parameters.cv_q.shape[0]
    transition = np.zeros((batch_size, 3, STATE_DIM, STATE_DIM), dtype=float)
    process = np.zeros_like(transition)

    _fill_cv_model(transition[:, 0], process[:, 0], parameters.cv_q)
    _fill_ct_model(
        transition[:, 1],
        process[:, 1],
        parameters.ct_turn_rate,
        parameters.ct_q,
    )
    _fill_singer_model(
        transition[:, 2],
        process[:, 2],
        parameters.singer_alpha,
        parameters.singer_sigma,
    )
    process = 0.5 * (process + np.swapaxes(process, -1, -2))
    diagonal = np.arange(STATE_DIM)
    process[:, :, diagonal, diagonal] += 1.0e-9
    return transition, process


def _fill_cv_model(f: np.ndarray, q: np.ndarray, q_xyz: np.ndarray) -> None:
    for axis, start in enumerate((0, 3, 6)):
        f[:, start, start] = 1.0
        f[:, start, start + 1] = 1.0
        f[:, start + 1, start + 1] = 1.0
        f[:, start + 2, start + 2] = 0.05
        value = q_xyz[:, axis]
        q[:, start, start] = value / 3.0
        q[:, start, start + 1] = value / 2.0
        q[:, start + 1, start] = value / 2.0
        q[:, start + 1, start + 1] = value
        q[:, start + 2, start + 2] = value


def _fill_ct_model(
    f: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    q_xyz: np.ndarray,
) -> None:
    sine = np.sin(omega)
    cosine = np.cos(omega)
    small = np.abs(omega) < 1.0e-8
    sine_over_omega = np.divide(
        sine,
        omega,
        out=np.ones_like(omega),
        where=~small,
    )
    one_minus_cosine_over_omega = np.divide(
        1.0 - cosine,
        omega,
        out=np.zeros_like(omega),
        where=~small,
    )
    f[:, 0, 0] = 1.0
    f[:, 0, 1] = sine_over_omega
    f[:, 0, 4] = -one_minus_cosine_over_omega
    f[:, 1, 1] = cosine
    f[:, 1, 4] = -sine
    f[:, 2, 1] = cosine - 1.0
    f[:, 2, 4] = -sine
    f[:, 3, 3] = 1.0
    f[:, 3, 1] = one_minus_cosine_over_omega
    f[:, 3, 4] = sine_over_omega
    f[:, 4, 1] = sine
    f[:, 4, 4] = cosine
    f[:, 5, 1] = sine
    f[:, 5, 4] = cosine - 1.0

    horizontal_q = np.mean(q_xyz[:, :2], axis=1)
    indices = np.array([0, 1, 3, 4])
    reference = np.array(
        [
            [1.0 / 3.0, 0.0, 1.0 / 2.0, 0.0],
            [0.0, 1.0 / 3.0, 0.0, 1.0 / 2.0],
            [1.0 / 2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0 / 2.0, 0.0, 1.0],
        ]
    )
    permuted = reference[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]
    q[:, indices[:, None], indices[None, :]] = (
        horizontal_q[:, None, None] * permuted[None, :, :]
    )
    q[:, 2, 2] = horizontal_q
    q[:, 5, 5] = horizontal_q

    vertical_f = np.zeros((f.shape[0], STATE_DIM, STATE_DIM), dtype=float)
    vertical_q = np.zeros_like(vertical_f)
    _fill_cv_model(vertical_f, vertical_q, q_xyz)
    f[:, 6:9, 6:9] = vertical_f[:, 6:9, 6:9]
    q[:, 6:9, 6:9] = vertical_q[:, 6:9, 6:9]


def _fill_singer_model(
    f: np.ndarray,
    q: np.ndarray,
    alpha_xyz: np.ndarray,
    sigma_xyz: np.ndarray,
) -> None:
    for axis, start in enumerate((0, 3, 6)):
        alpha = np.maximum(alpha_xyz[:, axis], 1.0e-6)
        sigma = np.maximum(np.abs(sigma_xyz[:, axis]), 1.0e-6)
        exponential = np.exp(-alpha)
        exponential2 = np.exp(-2.0 * alpha)
        f[:, start, start] = 1.0
        f[:, start, start + 1] = 1.0
        f[:, start, start + 2] = (alpha - 1.0 + exponential) / np.square(alpha)
        f[:, start + 1, start + 1] = 1.0
        f[:, start + 1, start + 2] = (1.0 - exponential) / alpha
        f[:, start + 2, start + 2] = exponential

        q11 = (
            2.0 * alpha**3
            - 6.0 * alpha**2
            + 6.0 * alpha
            + 3.0
            - 12.0 * alpha * exponential
            - 3.0 * exponential2
        ) / (6.0 * alpha**5)
        q12 = (
            alpha**2
            - 2.0 * alpha
            + 1.0
            - 2.0 * (1.0 - alpha) * exponential
            + exponential2
        ) / (2.0 * alpha**4)
        q22 = (
            2.0 * alpha - 3.0 + 4.0 * exponential - exponential2
        ) / (2.0 * alpha**3)
        q13 = (
            1.0 - 2.0 * alpha * exponential - exponential2
        ) / (2.0 * alpha**3)
        q23 = np.square(1.0 - exponential) / (2.0 * alpha**2)
        q33 = (1.0 - exponential2) / (2.0 * alpha)
        scale = 2.0 * alpha * np.square(sigma)
        values = (
            (0, 0, q11),
            (0, 1, q12),
            (1, 0, q12),
            (1, 1, q22),
            (0, 2, q13),
            (2, 0, q13),
            (1, 2, q23),
            (2, 1, q23),
            (2, 2, q33),
        )
        for row, column, value in values:
            q[:, start + row, start + column] = scale * value


def _initial_covariance(
    batch_size: int,
    scale: float,
    model_count: int | None = None,
) -> np.ndarray:
    count = len(MODEL_NAMES) if model_count is None else int(model_count)
    diagonal = np.tile(
        np.array(
            [
                scale,
                scale * 25.0,
                scale * 100.0,
                scale,
                scale * 25.0,
                scale * 100.0,
                scale,
                scale * 25.0,
                scale * 100.0,
            ]
        ),
        (batch_size, count, 1),
    )
    covariance = np.zeros(
        (batch_size, count, STATE_DIM, STATE_DIM),
        dtype=float,
    )
    indices = np.arange(STATE_DIM)
    covariance[:, :, indices, indices] = diagonal
    return covariance


def _imm_predict(
    states: np.ndarray,
    covariance: np.ndarray,
    probabilities: np.ndarray,
    model_transition: np.ndarray,
    dynamics: np.ndarray,
    process_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_probabilities = np.einsum(
        "bij,bi->bj", model_transition, probabilities
    )
    predicted_probabilities = np.clip(predicted_probabilities, 1.0e-12, None)
    mixing_weights = (
        model_transition * probabilities[:, :, None]
    ) / predicted_probabilities[:, None, :]
    mixed_states = np.einsum("bst,bsd->btd", mixing_weights, states)
    deltas = states[:, :, None, :] - mixed_states[:, None, :, :]
    mixed_covariance = np.einsum(
        "bst,bsij->btij", mixing_weights, covariance
    ) + np.einsum(
        "bst,bsti,bstj->btij", mixing_weights, deltas, deltas
    )
    predicted_states = np.einsum("bmij,bmj->bmi", dynamics, mixed_states)
    predicted_covariance = (
        dynamics
        @ mixed_covariance
        @ np.swapaxes(dynamics, -1, -2)
        + process_covariance
    )
    predicted_covariance = 0.5 * (
        predicted_covariance + np.swapaxes(predicted_covariance, -1, -2)
    )
    predicted_probabilities /= predicted_probabilities.sum(axis=1, keepdims=True)
    return predicted_states, predicted_covariance, predicted_probabilities


def _measurement_update(
    states: np.ndarray,
    covariance: np.ndarray,
    probabilities: np.ndarray,
    measurements: np.ndarray,
    update_mask: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.flatnonzero(update_mask)
    if active.size == 0:
        return states, covariance, probabilities
    updated_states = states.copy()
    updated_covariance = covariance.copy()
    model_count = states.shape[1]
    log_likelihoods = np.zeros((active.size, model_count), dtype=float)
    for model_index in range(model_count):
        model_states = states[active, model_index]
        model_covariance = covariance[active, model_index]
        innovation = measurements[active] - model_states[:, POSITION_INDICES]
        innovation_covariance = (
            model_covariance[:, POSITION_INDICES][:, :, POSITION_INDICES]
            + measurement_covariance[None, :, :]
        )
        inverse = np.linalg.inv(innovation_covariance)
        gain = (
            model_covariance[:, :, POSITION_INDICES] @ inverse
        )
        updated_states[active, model_index] = (
            model_states + np.einsum("bij,bj->bi", gain, innovation)
        )
        reduction = gain @ innovation_covariance @ np.swapaxes(gain, -1, -2)
        model_updated_covariance = model_covariance - reduction
        updated_covariance[active, model_index] = 0.5 * (
            model_updated_covariance
            + np.swapaxes(model_updated_covariance, -1, -2)
        )
        _, log_determinant = np.linalg.slogdet(innovation_covariance)
        quadratic = np.einsum(
            "bi,bij,bj->b", innovation, inverse, innovation
        )
        log_likelihoods[:, model_index] = -0.5 * (
            quadratic + log_determinant + 3.0 * np.log(2.0 * np.pi)
        )

    log_weights = np.log(np.clip(probabilities[active], 1.0e-300, None)) + log_likelihoods
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    updated_probabilities = probabilities.copy()
    updated_probabilities[active] = weights / weights.sum(axis=1, keepdims=True)
    return updated_states, updated_covariance, updated_probabilities


def _imm_forecast_means(
    states: np.ndarray,
    probabilities: np.ndarray,
    model_transition: np.ndarray,
    dynamics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_probabilities = np.einsum(
        "bij,bi->bj", model_transition, probabilities
    )
    predicted_probabilities = np.clip(predicted_probabilities, 1.0e-12, None)
    mixing_weights = (
        model_transition * probabilities[:, :, None]
    ) / predicted_probabilities[:, None, :]
    mixed_states = np.einsum("bst,bsd->btd", mixing_weights, states)
    predicted_states = np.einsum("bmij,bmj->bmi", dynamics, mixed_states)
    predicted_probabilities /= predicted_probabilities.sum(axis=1, keepdims=True)
    return predicted_states, predicted_probabilities
