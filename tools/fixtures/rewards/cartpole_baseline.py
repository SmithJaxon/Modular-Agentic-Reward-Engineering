"""
Summary: Baseline deterministic reward function fixture for CartPole-like tests.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations


def compute_reward(position_error: float, angle_error: float) -> float:
    """
    Compute a simple baseline reward using position and angle penalties.

    Args:
        position_error: Absolute cart position deviation from center.
        angle_error: Absolute pole angle deviation from vertical.

    Returns:
        Reward score where larger values are better.
    """
    return 1.0 - (0.1 * abs(position_error)) - (0.9 * abs(angle_error))
