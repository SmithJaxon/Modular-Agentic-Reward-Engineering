"""
Summary: RewardLab-compatible adapter for an external Humanoid reward.
Created: 2026-04-21
Last Updated: 2026-04-21
"""

from __future__ import annotations

import math
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort scalar conversion with safe fallback."""

    try:
        return float(value)
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            return float(value.item())
    except Exception:
        pass
    return default


def _to_vector(value: Any, size: int) -> list[float] | None:
    """Convert list-like values to a fixed-size vector when possible."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) < size:
            return None
        return [_to_float(value[index]) for index in range(size)]
    try:
        if hasattr(value, "tolist"):
            seq = value.tolist()
            if isinstance(seq, list) and len(seq) >= size:
                return [_to_float(seq[index]) for index in range(size)]
    except Exception:
        pass
    return None


def _info_scalar(info: dict[str, Any] | None, keys: tuple[str, ...]) -> float | None:
    if not info:
        return None
    for key in keys:
        if key in info:
            return _to_float(info.get(key))
    return None


def reward(
    env_reward: float = 0.0,
    environment_reward: float | None = None,
    observation: Any = None,
    next_observation: Any = None,
    action: Any = None,
    info: dict[str, Any] | None = None,
    **kwargs: object,
) -> float:
    """Scalar Humanoid shaping adapted from the external tensor reward."""

    del kwargs
    base = _to_float(environment_reward if environment_reward is not None else env_reward, 0.0)

    progress = _info_scalar(info, ("progress", "potential_delta"))
    potentials = _info_scalar(info, ("potentials", "potential"))
    prev_potentials = _info_scalar(info, ("prev_potentials", "previous_potentials"))
    if progress is None and potentials is not None and prev_potentials is not None:
        progress = potentials - prev_potentials

    speed = _info_scalar(info, ("horiz_speed", "speed", "velocity_towards_target"))
    up_signal = _info_scalar(info, ("up_reward", "up_proj", "up_z"))
    heading_signal = _info_scalar(info, ("heading_reward", "heading_proj", "heading_z"))

    obs_value = next_observation if next_observation is not None else observation
    obs = _to_vector(obs_value, 12)
    if obs is not None:
        # Humanoid observations typically contain up/heading projections around these indices.
        if up_signal is None:
            up_signal = _to_float(obs[10], 0.0)
        if heading_signal is None:
            heading_signal = _to_float(obs[11], 0.0)

    # If no usable signals are exposed by the runtime payload, preserve environment reward.
    if progress is None and speed is None and up_signal is None and heading_signal is None:
        return float(base)

    progress_value = _to_float(progress, 0.0)
    speed_value = _to_float(speed, 0.0)
    up_value = max(0.0, min(1.0, _to_float(up_signal, 0.0)))
    heading_value = max(0.0, min(1.0, _to_float(heading_signal, 0.0)))

    action_penalty = 0.0
    if action is not None:
        if isinstance(action, (list, tuple)):
            action_penalty = sum((_to_float(value) ** 2.0) for value in action)
        else:
            action_penalty = _to_float(action, 0.0) ** 2.0

    progress_reward = math.exp(min(progress_value, 10.0))
    speed_reward = math.exp(min(speed_value, 10.0))
    upright_reward = math.exp(min(0.5 * (up_value + heading_value), 10.0))

    shaped = 2.0 * progress_reward + 1.5 * speed_reward + 0.5 * upright_reward - 0.01 * action_penalty
    shaped = max(0.0, shaped)
    if math.isfinite(shaped):
        return float(shaped)
    return float(base)

