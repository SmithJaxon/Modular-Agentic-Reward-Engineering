"""
Summary: Deterministic baseline reward fixture for CartPole validation.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations


def compute_reward(
    cart_position: float,
    pole_angle_radians: float,
    angular_velocity: float,
    terminated: bool,
) -> float:
    """Return a simple deterministic reward for fixture-based validation."""
    if terminated:
        return -10.0

    centered_bonus = max(0.0, 1.0 - abs(cart_position))
    upright_bonus = max(0.0, 1.0 - abs(pole_angle_radians) * 2.5)
    stability_penalty = min(abs(angular_velocity) * 0.1, 1.0)
    return round(centered_bonus + upright_bonus - stability_penalty, 6)
