"""
Summary: Candidate ranking and best-candidate selection policy for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Sequence

from rewardlab.schemas.reward_candidate import RewardCandidate


class CandidateSelectionPolicy:
    """Rank reward candidates using the currently available session signals."""

    def rank_candidates(self, candidates: Sequence[RewardCandidate]) -> list[RewardCandidate]:
        """Return candidates sorted from best to worst using deterministic rules."""

        return sorted(
            candidates,
            key=lambda candidate: (
                (
                    candidate.aggregate_score
                    if candidate.aggregate_score is not None
                    else float("-inf")
                ),
                candidate.iteration_index,
            ),
            reverse=True,
        )

    def select_best_candidate(self, candidates: Sequence[RewardCandidate]) -> RewardCandidate:
        """Return the best candidate from a non-empty collection."""

        ranked = self.rank_candidates(candidates)
        if not ranked:
            raise ValueError("at least one candidate is required")
        return ranked[0]
