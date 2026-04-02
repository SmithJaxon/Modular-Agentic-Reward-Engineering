"""
Summary: Baseline multi-signal selection policy for choosing best reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass


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
