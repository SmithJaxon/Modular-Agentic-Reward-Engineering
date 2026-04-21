"""
Summary: Sparse reward fixture for Isaac Gym Humanoid.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def reward(
    x_velocity: float | None = None,
    cur_dist: float | None = None,
    prev_dist: float | None = None,
    **kwargs: object,
) -> float:
    """Approximate Eureka's sparse progress metric for Humanoid."""

    del kwargs
    if cur_dist is not None and prev_dist is not None:
        return float(cur_dist) - float(prev_dist)
    if x_velocity is not None:
        return float(x_velocity)
    return 0.0

