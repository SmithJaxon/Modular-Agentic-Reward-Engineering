"""
Summary: Human baseline reward fixture for Isaac Gym Cartpole.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def reward(
    env_reward: float,
    **kwargs: object,
) -> float:
    """Use Isaac Gym's built-in task reward as the human baseline proxy."""

    del kwargs
    return float(env_reward)

