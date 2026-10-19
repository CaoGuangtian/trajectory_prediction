"""Research-facing API for the causal gated motion IMM model."""

from trajectory_prediction.models.imm_kalman import (
    ImmBatchParameters,
    ImmKalmanConfig,
    ImmKalmanSource,
    ImmPredictionBatch,
    evaluate_imm_kalman,
    load_imm_kalman_config,
    predict_imm_batch,
)

GatedMotionImmConfig = ImmKalmanConfig
GatedMotionImmSource = ImmKalmanSource
GatedMotionPredictionBatch = ImmPredictionBatch
evaluate_gated_motion_imm = evaluate_imm_kalman
load_gated_motion_imm_config = load_imm_kalman_config
predict_gated_motion_imm_batch = predict_imm_batch

__all__ = [
    "GatedMotionImmConfig",
    "GatedMotionImmSource",
    "GatedMotionPredictionBatch",
    "ImmBatchParameters",
    "evaluate_gated_motion_imm",
    "load_gated_motion_imm_config",
    "predict_gated_motion_imm_batch",
]
