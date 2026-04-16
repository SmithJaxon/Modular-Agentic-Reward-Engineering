"""
Summary: Cost-efficiency helpers for stop-guidance decisions.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CostEfficiencyAssessment:
    """
    Describe observed gain-per-cost against a configured minimum threshold.
    """

    score_gain: float
    api_cost_usd: float
    gain_per_1k_usd: float | None
    minimum_gain_per_1k_usd: float
    is_efficient: bool


def assess_cost_efficiency(
    *,
    score_gain: float,
    api_cost_usd: float,
    minimum_gain_per_1k_usd: float,
) -> CostEfficiencyAssessment:
    """
    Compute whether observed score gain per 1k USD meets the configured threshold.
    """
    if minimum_gain_per_1k_usd <= 0.0:
        return CostEfficiencyAssessment(
            score_gain=score_gain,
            api_cost_usd=api_cost_usd,
            gain_per_1k_usd=None,
            minimum_gain_per_1k_usd=minimum_gain_per_1k_usd,
            is_efficient=True,
        )
    if api_cost_usd <= 0.0:
        return CostEfficiencyAssessment(
            score_gain=score_gain,
            api_cost_usd=api_cost_usd,
            gain_per_1k_usd=None,
            minimum_gain_per_1k_usd=minimum_gain_per_1k_usd,
            is_efficient=True,
        )
    gain_per_1k = (score_gain / api_cost_usd) * 1000.0
    return CostEfficiencyAssessment(
        score_gain=score_gain,
        api_cost_usd=api_cost_usd,
        gain_per_1k_usd=gain_per_1k,
        minimum_gain_per_1k_usd=minimum_gain_per_1k_usd,
        is_efficient=gain_per_1k >= minimum_gain_per_1k_usd,
    )
