"""
Summary: Robustness risk analysis helpers for RewardLab candidate evaluation.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment


class RiskAnalyzer:
    """Assess robustness degradation and assign a reward-hacking risk level."""

    def assess_candidate(
        self,
        *,
        candidate: RewardCandidate,
        runs: list[ExperimentRun],
    ) -> RobustnessAssessment:
        """Summarize robustness degradation for a candidate across probe variants."""

        if not runs:
            raise ValueError("at least one robustness run is required")

        primary_score = candidate.aggregate_score or 0.0
        worst_variant_score = min(
            float(run.metrics.get("total_reward", 0.0)) for run in runs
        )
        degradation_ratio = _compute_degradation_ratio(primary_score, worst_variant_score)
        risk_level = _classify_risk(degradation_ratio)
        return RobustnessAssessment(
            assessment_id=f"{candidate.candidate_id}-robustness",
            candidate_id=candidate.candidate_id,
            variant_count=len(runs),
            degradation_ratio=degradation_ratio,
            risk_level=risk_level,
            risk_notes=(
                f"Worst robustness score {worst_variant_score:.3f} versus "
                f"primary score {primary_score:.3f}."
            ),
        )


def _compute_degradation_ratio(primary_score: float, worst_variant_score: float) -> float:
    """Return the relative degradation from the primary score to the worst variant."""

    if primary_score <= 0:
        return 0.0 if worst_variant_score >= 0 else 1.0
    return round(max(0.0, (primary_score - worst_variant_score) / primary_score), 4)


def _classify_risk(degradation_ratio: float) -> RiskLevel:
    """Map a degradation ratio to a deterministic risk bucket."""

    if degradation_ratio >= 0.5:
        return RiskLevel.HIGH
    if degradation_ratio >= 0.2:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
