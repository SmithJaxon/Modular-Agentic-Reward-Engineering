"""
Summary: Sparse reward fixture for Isaac Gym Cartpole.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def reward(
    terminated: bool,
    **kwargs: object,
) -> float:
    """Return per-step survival signal used by Eureka's Cartpole task fitness."""

    del kwargs
    return 0.0 if terminated else 1.0

