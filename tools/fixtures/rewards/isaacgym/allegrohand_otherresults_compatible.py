"""
Summary: RewardLab-compatible adapter for an external AllegroHand reward.
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
    """Convert list-like values to a fixed-size float vector when possible."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) < size:
            return None
        return [_to_float(value[idx]) for idx in range(size)]
    try:
        if hasattr(value, "tolist"):
            seq = value.tolist()
            if isinstance(seq, list) and len(seq) >= size:
                return [_to_float(seq[idx]) for idx in range(size)]
    except Exception:
        pass
    return None


def _get_info_vector(info: dict[str, Any] | None, keys: tuple[str, ...], size: int) -> list[float] | None:
    """Return the first matching vector value from an info mapping."""

    if not info:
        return None
    for key in keys:
        if key in info:
            vector = _to_vector(info.get(key), size)
            if vector is not None:
                return vector
    return None


def _quat_conjugate(quat: list[float]) -> list[float]:
    return [-quat[0], -quat[1], -quat[2], quat[3]]


def _quat_mul(lhs: list[float], rhs: list[float]) -> list[float]:
    """Quaternion multiply with [x, y, z, w] layout."""

    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def reward(
    env_reward: float = 0.0,
    environment_reward: float | None = None,
    observation: Any = None,
    next_observation: Any = None,
    action: Any = None,
    info: dict[str, Any] | None = None,
    **kwargs: object,
) -> float:
    """Scalar AllegroHand shaping derived from the external reward function."""

    del kwargs
    base = _to_float(environment_reward if environment_reward is not None else env_reward, 0.0)

    obs_value = next_observation if next_observation is not None else observation
    obs = _to_vector(obs_value, 72)

    object_rot = _get_info_vector(info, ("object_rot", "object_quat"), 4)
    goal_rot = _get_info_vector(info, ("goal_rot", "target_rot", "goal_quat"), 4)
    object_angvel = _get_info_vector(info, ("object_angvel", "object_angular_velocity"), 3)
    actions = _get_info_vector(info, ("actions", "action"), 16)

    # Allegro full observation mode packs these slices. If info lacks explicit keys, use obs fallback.
    if obs is not None:
        if object_rot is None:
            object_rot = obs[35:39]
        if goal_rot is None:
            goal_rot = obs[48:52]
        if object_angvel is None:
            object_angvel = obs[42:45]
        if actions is None:
            actions = obs[56:72]

    if actions is None:
        actions = _to_vector(action, 16)

    if object_rot is None or goal_rot is None or object_angvel is None:
        return float(base)

    rel_quat = _quat_mul(object_rot, _quat_conjugate(goal_rot))
    quat_err = 1.0 - abs(_to_float(rel_quat[3], 1.0))
    orientation_reward = math.exp(-quat_err / 0.25)

    angvel_mag = math.sqrt(sum((_to_float(value) ** 2.0) for value in object_angvel))
    spin_reward = math.tanh(angvel_mag / 1.0)

    action_sq = 0.0
    if actions is not None:
        action_sq = sum((_to_float(value) ** 2.0) for value in actions)
    action_penalty = math.exp(-action_sq / 10.0)

    shaped = 2.0 * orientation_reward + 0.5 * spin_reward + 0.1 * action_penalty
    if math.isfinite(shaped):
        return float(shaped)
    return float(base)

