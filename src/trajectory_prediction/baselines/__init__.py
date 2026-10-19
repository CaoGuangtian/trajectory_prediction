"""Reference baselines for trajectory-prediction experiments."""

from .constant_velocity import (
    ConstantVelocityConfig,
    ConstantVelocitySource,
    evaluate_constant_velocity,
    load_constant_velocity_config,
)
from .position_hold import (
    PositionHoldConfig,
    PositionHoldSource,
    evaluate_position_hold,
    load_position_hold_config,
)

__all__ = [
    "ConstantVelocityConfig",
    "ConstantVelocitySource",
    "PositionHoldConfig",
    "PositionHoldSource",
    "evaluate_constant_velocity",
    "evaluate_position_hold",
    "load_constant_velocity_config",
    "load_position_hold_config",
]
