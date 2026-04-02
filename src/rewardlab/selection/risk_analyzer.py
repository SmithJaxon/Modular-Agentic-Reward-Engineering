"""
Summary: Robustness risk analysis and tradeoff rationale helpers for candidate selection.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment


@dataclass(slots=True, frozen=True)
class RiskAnalysisResult:
    """
    Bundle a robustness assessment with policy-facing tradeoff signals.
    """

    assessment: RobustnessAssessment
    robustness_bonus: float
    tradeoff_rationale: str | None
    minor_robustness_risk_accepted: bool


class RiskAnalyzer:
    """
    Convert robustness probe outputs into risk levels and selection tradeoffs.
    """

    def analyze(
        self,
        candidate_id: str,
        primary_score: float,
        robustness_runs: list[ExperimentRun],
    ) -> RiskAnalysisResult:
        """
        Summarize probe outcomes into a normalized assessment and selection signals.

        Args:
            candidate_id: Candidate identifier tied to the probe results.
            primary_score: Primary non-robustness score for the candidate.
            robustness_runs: Completed robustness runs for probe variants.

        Returns:
            Risk analysis result containing assessment and selection modifiers.
        """
        if not robustness_runs:
            raise ValueError("robustness analysis requires at least one probe run")

        scores = [float(run.metrics["score"]) for run in robustness_runs]
        mean_score = sum(scores) / len(scores)
        degradation_ratio = max(0.0, (primary_score - mean_score) / max(primary_score, 0.001))
        worst_run = min(robustness_runs, key=lambda run: float(run.metrics["score"]))
        worst_delta = primary_score - float(worst_run.metrics["score"])

        if degradation_ratio >= 0.30 or worst_delta >= 0.25:
            risk_level = RiskLevel.HIGH
            risk_notes = (
                "High reward-hacking risk: robustness probes materially degraded under "
                f"{worst_run.variant_label} (degradation={degradation_ratio:.3f})."
            )
            robustness_bonus = -0.18
            tradeoff_rationale = None
        elif degradation_ratio >= 0.15 or worst_delta >= 0.15:
            risk_level = RiskLevel.MEDIUM
            risk_notes = (
                "Moderate degradation detected: candidate shows minor robustness risk "
                f"across {len(robustness_runs)} variants."
            )
            robustness_bonus = -0.03
            tradeoff_rationale = (
                "primary performance remained strong while probe degradation stayed "
                "within discretionary bounds"
            )
        else:
            risk_level = RiskLevel.LOW
            risk_notes = (
                "Low robustness risk: probe scores remained stable across "
                f"{len(robustness_runs)} variants."
            )
            robustness_bonus = 0.05
            tradeoff_rationale = None

        assessment = RobustnessAssessment(
            assessment_id=f"assess-{uuid4().hex[:12]}",
            candidate_id=candidate_id,
            variant_count=len(robustness_runs),
            degradation_ratio=round(degradation_ratio, 4),
            risk_level=risk_level,
            risk_notes=risk_notes,
            created_at=datetime.now(UTC).isoformat(),
        )
        return RiskAnalysisResult(
            assessment=assessment,
            robustness_bonus=robustness_bonus,
            tradeoff_rationale=tradeoff_rationale,
            minor_robustness_risk_accepted=risk_level is RiskLevel.MEDIUM,
        )
