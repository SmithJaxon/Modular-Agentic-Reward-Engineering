"""
Summary: Sparse reward fixture for Gymnasium Humanoid forward-locomotion comparisons.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def reward(
    x_velocity: float,
    terminated: bool,
    truncated: bool,
    **kwargs: object,
) -> float:
    """Sparse forward-progress signal with no dense shaping terms."""

    del kwargs
    if terminated or truncated:
        return 0.0
    return 1.0 if float(x_velocity) > 0.0 else 0.0
