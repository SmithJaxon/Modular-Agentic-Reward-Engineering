"""
Summary: Baseline reward fixture for Gymnasium Humanoid forward-locomotion runs.
Created: 2026-04-06
Last Updated: 2026-04-16
"""

from __future__ import annotations


def reward(
    observation: object,
    x_velocity: float,
    reward_alive: float,
    reward_quadctrl: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """Return a simple forward-velocity objective with alive/control shaping."""

    if terminated or truncated:
        return float(reward_alive) - float(reward_quadctrl)

    del observation
    velocity_bonus = float(x_velocity)
    upright_bonus = float(reward_alive)
    control_penalty = float(reward_quadctrl)
    return velocity_bonus + upright_bonus - control_penalty
