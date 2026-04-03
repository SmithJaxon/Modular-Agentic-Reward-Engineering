"""
Summary: Baseline multi-signal selection policy for choosing best reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass

from rewardlab.schemas.robustness_assessment import RiskLevel


@dataclass(slots=True, frozen=True)
class CandidateSignal:
    """
    Represent scored signals used by the baseline candidate selection policy.
    """

    candidate_id: str
    primary_performance: float
    robustness_bonus: float = 0.0
    human_feedback_bonus: float = 0.0
    peer_feedback_bonus: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    tradeoff_rationale: str | None = None

    @property
    def aggregate_score(self) -> float:
        """
        Compute aggregate score across all baseline selection signals.
        """
        return (
            self.primary_performance
            + self.robustness_bonus
            + self.human_feedback_bonus
            + self.peer_feedback_bonus
        )

    @property
    def minor_robustness_risk_accepted(self) -> bool:
        """
        Report whether the signal represents an accepted minor robustness tradeoff.
        """
        return self.risk_level is RiskLevel.MEDIUM and self.tradeoff_rationale is not None

    @property
    def has_conflicting_feedback(self) -> bool:
        """
        Report whether human and peer feedback bonuses point in opposite directions.
        """
        return self.human_feedback_bonus * self.peer_feedback_bonus < 0


@dataclass(slots=True, frozen=True)
class SelectionOutcome:
    """
    Represent the policy decision for the currently best candidate.
    """

    selected_signal: CandidateSignal
    selection_summary: str


def choose_best_signal(candidates: list[CandidateSignal]) -> CandidateSignal:
    """
    Select the highest-scoring candidate signal entry.

    Args:
        candidates: Candidate signal values.

    Returns:
        Highest aggregate score candidate.

    Raises:
        ValueError: If candidate list is empty.
    """
    if not candidates:
        raise ValueError("cannot choose best candidate from empty list")
    return max(candidates, key=lambda candidate: candidate.aggregate_score)


def select_candidate(candidates: list[CandidateSignal]) -> SelectionOutcome:
    """
    Choose the current best candidate and capture an auditable selection summary.

    Args:
        candidates: Candidate signal values.

    Returns:
        Policy outcome containing the selected signal and rationale summary.
    """
    best = choose_best_signal(candidates)
    summary = (
        f"Selected candidate {best.candidate_id} with aggregate score "
        f"{best.aggregate_score:.3f} from primary performance {best.primary_performance:.3f} "
        f"and robustness bonus {best.robustness_bonus:+.3f}."
    )
    if best.human_feedback_bonus or best.peer_feedback_bonus:
        summary = (
            f"{summary} Feedback bonuses human {best.human_feedback_bonus:+.3f}, "
            f"peer {best.peer_feedback_bonus:+.3f}."
        )
    if best.has_conflicting_feedback:
        summary = f"{summary} Conflicting human and peer feedback remained under review."
    if best.minor_robustness_risk_accepted:
        summary = f"{summary} Minor robustness risk accepted: {best.tradeoff_rationale}"
    elif best.risk_level is RiskLevel.HIGH:
        summary = f"{summary} High robustness risk remained visible during selection."
    return SelectionOutcome(selected_signal=best, selection_summary=summary)
