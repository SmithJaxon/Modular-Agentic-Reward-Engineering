"""
Summary: Baseline reward fixture for Humanoid-v4 PPO sessions.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations


def compute_reward(observation, action, next_observation, env_reward, terminated, truncated, info):
    """
    Blend environment reward with lightweight stability terms for a safe baseline.
    """
    _ = action
    _ = terminated
    _ = truncated
    _ = info

    next_obs = next_observation
    torso_height = float(next_obs[0]) if len(next_obs) > 0 else 0.0
    torso_vertical = float(next_obs[1]) if len(next_obs) > 1 else 1.0
    stability_bonus = max(0.0, min(1.0, torso_vertical))
    height_bonus = max(0.0, min(1.0, torso_height / 1.4))

    reward = 0.9 * float(env_reward) + 0.07 * stability_bonus + 0.03 * height_bonus
    return reward, {
        "env_reward": float(env_reward),
        "stability_bonus": stability_bonus,
        "height_bonus": height_bonus,
    }
