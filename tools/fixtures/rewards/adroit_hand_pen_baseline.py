"""
Summary: Baseline reward fixture for AdroitHandPen-v1 PPO sessions.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

import numpy as np


def compute_reward(observation, action, next_observation, env_reward, terminated, truncated, info):
    """
    Preserve task reward while lightly penalizing excessive control effort.
    """
    _ = observation
    _ = next_observation
    _ = terminated
    _ = truncated
    _ = info

    action_array = np.asarray(action, dtype=np.float32)
    control_cost = float(np.mean(np.square(action_array))) if action_array.size else 0.0
    reward = float(env_reward) - (0.002 * control_cost)
    return reward, {
        "env_reward": float(env_reward),
        "control_cost": control_cost,
    }
