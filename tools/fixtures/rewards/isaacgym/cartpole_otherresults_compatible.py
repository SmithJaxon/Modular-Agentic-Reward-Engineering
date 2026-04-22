"""
Summary: RewardLab-compatible adapter for an external Cartpole reward.
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


def _extract_observation(
    *,
    state: Any = None,
    observation: Any = None,
    next_observation: Any = None,
) -> list[float] | None:
    """Extract a 4D Cartpole observation [x, x_dot, theta, theta_dot]."""

    source = next_observation if next_observation is not None else observation
    if source is None:
        source = state
    if source is None:
        return None
    if isinstance(source, (list, tuple)) and len(source) >= 4:
        return [_to_float(source[index]) for index in range(4)]
    try:
        if hasattr(source, "tolist"):
            seq = source.tolist()
            if isinstance(seq, list) and len(seq) >= 4:
                return [_to_float(seq[index]) for index in range(4)]
    except Exception:
        pass
    return None


def reward(
    env_reward: float = 0.0,
    environment_reward: float | None = None,
    state: Any = None,
    observation: Any = None,
    next_observation: Any = None,
    **kwargs: object,
) -> float:
    """Cartpole scalar reward based on the external tensor formula."""

    del kwargs
    base = _to_float(environment_reward if environment_reward is not None else env_reward, 0.0)
    obs = _extract_observation(state=state, observation=observation, next_observation=next_observation)
    if obs is None:
        return float(base)

    cart_pos, cart_vel, pole_angle, pole_ang_vel = obs

    # Matches the external compute_reward temperature-weighted shaping.
    upright_reward = math.exp(-(pole_angle * pole_angle) / (2.0 * 0.5 * 0.5))
    center_reward = math.exp(-(cart_pos * cart_pos) / (2.0 * 1.0 * 1.0))
    cart_vel_reward = math.exp(-(cart_vel * cart_vel) / (2.0 * 1.0 * 1.0))
    ang_vel_reward = math.exp(-(pole_ang_vel * pole_ang_vel) / (2.0 * 1.0 * 1.0))

    shaped = 1.5 * upright_reward + 0.3 * center_reward + 0.1 * cart_vel_reward + 0.1 * ang_vel_reward
    if math.isfinite(shaped):
        return float(shaped)
    return float(base)

