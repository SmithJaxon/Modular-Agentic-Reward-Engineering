"""
Summary: Sparse reward fixture for Isaac Gym AllegroHand.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def reward(
    consecutive_successes: float | None = None,
    rot_dist: float | None = None,
    **kwargs: object,
) -> float:
    """Approximate Eureka's sparse success metric for AllegroHand."""

    del kwargs
    if consecutive_successes is not None:
        return float(consecutive_successes)
    if rot_dist is not None:
        return 1.0 if float(rot_dist) < 0.1 else 0.0
    return 0.0

