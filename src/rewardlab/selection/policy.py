"""
Summary: Candidate ranking and best-candidate selection policy for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment


class CandidateSelectionPolicy:
    """Rank reward candidates using the currently available session signals."""

    def rank_candidates(
        self,
        candidates: Sequence[RewardCandidate],
        *,
        assessments: Mapping[str, RobustnessAssessment] | None = None,
    ) -> list[RewardCandidate]:
        """Return candidates sorted from best to worst using deterministic rules."""

        return sorted(
            candidates,
            key=lambda candidate: (
                (
                    self._effective_score(candidate, assessments=assessments)
                ),
                candidate.iteration_index,
            ),
            reverse=True,
        )

    def select_best_candidate(
        self,
        candidates: Sequence[RewardCandidate],
        *,
        assessments: Mapping[str, RobustnessAssessment] | None = None,
    ) -> RewardCandidate:
        """Return the best candidate from a non-empty collection."""

        ranked = self.rank_candidates(candidates, assessments=assessments)
        if not ranked:
            raise ValueError("at least one candidate is required")
        return ranked[0]

    def _effective_score(
        self,
        candidate: RewardCandidate,
        *,
        assessments: Mapping[str, RobustnessAssessment] | None = None,
    ) -> float:
        """Apply robustness penalties to the candidate's aggregate score when present."""

        base_score = (
            candidate.aggregate_score if candidate.aggregate_score is not None else float("-inf")
        )
        if assessments is None:
            return base_score
        assessment = assessments.get(candidate.candidate_id)
        if assessment is None:
            return base_score
        return base_score - _risk_penalty(assessment.risk_level)


def _risk_penalty(risk_level: RiskLevel) -> float:
    """Return the score penalty associated with a robustness risk bucket."""

    penalties = {
        RiskLevel.LOW: 0.0,
        RiskLevel.MEDIUM: 0.75,
        RiskLevel.HIGH: 2.0,
    }
    return penalties[risk_level]
