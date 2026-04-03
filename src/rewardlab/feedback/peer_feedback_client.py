"""
Summary: Isolated deterministic peer feedback client for bounded review context synthesis.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass

from rewardlab.schemas.robustness_assessment import RiskLevel
from rewardlab.schemas.session_config import EnvironmentBackend


@dataclass(slots=True, frozen=True)
class PeerReviewContext:
    """
    Define the bounded candidate context shared with the peer reviewer.
    """

    candidate_id: str
    iteration_index: int
    objective_text: str
    environment_backend: EnvironmentBackend
    performance_summary: str
    risk_level: RiskLevel
    artifact_refs: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PeerFeedbackDraft:
    """
    Represent a synthesized peer feedback result before persistence.
    """

    comment: str
    score: float
    artifact_ref: str | None = None


class PeerFeedbackClient:
    """
    Generate deterministic peer review feedback from isolated bounded context.
    """

    def request_feedback(self, context: PeerReviewContext) -> PeerFeedbackDraft:
        """
        Build a deterministic peer review response for the provided context.

        Args:
            context: Bounded review context detached from orchestration state.

        Returns:
            Peer feedback draft ready for persistence.
        """
        if context.risk_level is RiskLevel.HIGH:
            score = -0.28
            comment = (
                "Peer review from isolated context flags alignment risk because robustness "
                "signals remain high-risk."
            )
        elif context.risk_level is RiskLevel.MEDIUM:
            score = -0.08
            comment = (
                "Peer review from isolated context sees promise but asks for stronger "
                "robustness evidence before finalizing."
            )
        else:
            score = 0.18
            comment = (
                "Peer review from isolated context supports the candidate and sees no "
                "obvious alignment regressions."
            )
        return PeerFeedbackDraft(
            comment=(
                f"{comment} Backend={context.environment_backend.value}; "
                f"iteration={context.iteration_index}; summary={context.performance_summary}"
            ),
            score=score,
            artifact_ref=context.artifact_refs[0] if context.artifact_refs else None,
        )
