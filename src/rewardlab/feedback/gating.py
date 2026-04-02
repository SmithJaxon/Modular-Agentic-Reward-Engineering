"""
Summary: Feedback gate evaluation helpers for RewardLab final recommendations.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSourceType
from rewardlab.schemas.session_config import FeedbackGate


@dataclass(frozen=True)
class FeedbackGateResult:
    """Outcome of evaluating session feedback requirements."""

    satisfied: bool
    missing_sources: tuple[FeedbackSourceType, ...]
    conflict_detected: bool


class FeedbackGateEvaluator:
    """Evaluate feedback gating policies and detect conflicting feedback signals."""

    def evaluate(
        self,
        *,
        feedback_gate: FeedbackGate,
        feedback_entries: Sequence[FeedbackEntry],
    ) -> FeedbackGateResult:
        """Return the gate outcome for the supplied feedback collection."""

        has_human = any(
            entry.source_type == FeedbackSourceType.HUMAN for entry in feedback_entries
        )
        has_peer = any(
            entry.source_type == FeedbackSourceType.PEER for entry in feedback_entries
        )
        missing_sources: list[FeedbackSourceType] = []
        if feedback_gate == FeedbackGate.ONE_REQUIRED:
            satisfied = has_human or has_peer
            if not satisfied:
                missing_sources = [FeedbackSourceType.HUMAN, FeedbackSourceType.PEER]
        elif feedback_gate == FeedbackGate.BOTH_REQUIRED:
            satisfied = has_human and has_peer
            if not has_human:
                missing_sources.append(FeedbackSourceType.HUMAN)
            if not has_peer:
                missing_sources.append(FeedbackSourceType.PEER)
        else:
            satisfied = True

        return FeedbackGateResult(
            satisfied=satisfied,
            missing_sources=tuple(missing_sources),
            conflict_detected=self.detect_conflict(feedback_entries),
        )

    def detect_conflict(self, feedback_entries: Sequence[FeedbackEntry]) -> bool:
        """Return whether the available feedback shows a strong disagreement."""

        human_scores = [
            entry.score
            for entry in feedback_entries
            if entry.source_type == FeedbackSourceType.HUMAN and entry.score is not None
        ]
        peer_scores = [
            entry.score
            for entry in feedback_entries
            if entry.source_type == FeedbackSourceType.PEER and entry.score is not None
        ]
        if not human_scores or not peer_scores:
            return False
        return (max(peer_scores) - min(human_scores)) >= 0.5
