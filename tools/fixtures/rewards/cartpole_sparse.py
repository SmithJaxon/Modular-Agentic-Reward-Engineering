"""
Summary: Sparse reward fixture for CartPole comparison baselines.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations


def compute_reward(
    terminated: bool,
    **kwargs: object,
) -> float:
    """Return 1 per alive step and 0 on failure."""

    del kwargs
    return 0.0 if terminated else 1.0
