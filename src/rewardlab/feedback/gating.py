"""
Summary: Feedback summarization and gate evaluation helpers for candidate recommendations.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSource
from rewardlab.schemas.session_config import FeedbackGate


@dataclass(slots=True, frozen=True)
class CandidateFeedbackState:
    """
    Summarize persisted feedback signals for one candidate.
    """

    candidate_id: str
    human_feedback_count: int = 0
    peer_feedback_count: int = 0
    human_feedback_bonus: float = 0.0
    peer_feedback_bonus: float = 0.0
    feedback_count: int = 0
    summary: str = "no feedback recorded"

    @property
    def has_human_feedback(self) -> bool:
        """
        Report whether human review exists for the candidate.
        """
        return self.human_feedback_count > 0

    @property
    def has_peer_feedback(self) -> bool:
        """
        Report whether peer review exists for the candidate.
        """
        return self.peer_feedback_count > 0

    @property
    def has_conflicting_feedback(self) -> bool:
        """
        Report whether human and peer feedback point in opposite score directions.
        """
        return self.human_feedback_bonus * self.peer_feedback_bonus < 0


@dataclass(slots=True, frozen=True)
class FeedbackGateResult:
    """
    Capture whether a candidate satisfies the configured feedback gate.
    """

    gate: FeedbackGate
    satisfied: bool
    missing_sources: tuple[FeedbackSource, ...]
    message: str


def summarize_feedback_entries(
    candidate_id: str,
    entries: Iterable[FeedbackEntry],
) -> CandidateFeedbackState:
    """
    Aggregate feedback entries into policy-facing counts and bonus values.

    Args:
        candidate_id: Candidate identifier.
        entries: Persisted feedback entries for the candidate.

    Returns:
        Aggregated candidate feedback state.
    """
    items = list(entries)
    human_scores = [
        float(entry.score)
        for entry in items
        if entry.source_type is FeedbackSource.HUMAN and entry.score is not None
    ]
    peer_scores = [
        float(entry.score)
        for entry in items
        if entry.source_type is FeedbackSource.PEER and entry.score is not None
    ]
    human_count = sum(1 for entry in items if entry.source_type is FeedbackSource.HUMAN)
    peer_count = sum(1 for entry in items if entry.source_type is FeedbackSource.PEER)
    human_bonus = round(sum(human_scores) / len(human_scores), 4) if human_scores else 0.0
    peer_bonus = round(sum(peer_scores) / len(peer_scores), 4) if peer_scores else 0.0
    summary_parts: list[str] = []
    if human_count:
        summary_parts.append(f"human x{human_count} ({human_bonus:+.2f})")
    if peer_count:
        summary_parts.append(f"peer x{peer_count} ({peer_bonus:+.2f})")
    if not summary_parts:
        summary = "no feedback recorded"
    else:
        summary = ", ".join(summary_parts)
        if human_bonus * peer_bonus < 0:
            summary = f"{summary}; conflicting feedback detected"
        latest_comment = _clip_comment(items[-1].comment)
        if latest_comment:
            summary = f"{summary}; latest: {latest_comment}"
    return CandidateFeedbackState(
        candidate_id=candidate_id,
        human_feedback_count=human_count,
        peer_feedback_count=peer_count,
        human_feedback_bonus=human_bonus,
        peer_feedback_bonus=peer_bonus,
        feedback_count=len(items),
        summary=summary,
    )


def summarize_feedback_by_candidate(
    entries: Iterable[FeedbackEntry],
) -> dict[str, CandidateFeedbackState]:
    """
    Group feedback entries by candidate and summarize each bucket.

    Args:
        entries: Persisted feedback entries across one session.

    Returns:
        Mapping of candidate identifier to aggregated feedback state.
    """
    buckets: dict[str, list[FeedbackEntry]] = defaultdict(list)
    for entry in entries:
        buckets[entry.candidate_id].append(entry)
    return {
        candidate_id: summarize_feedback_entries(candidate_id, candidate_entries)
        for candidate_id, candidate_entries in buckets.items()
    }


def evaluate_feedback_gate(
    gate: FeedbackGate,
    feedback_state: CandidateFeedbackState,
) -> FeedbackGateResult:
    """
    Evaluate whether a candidate satisfies the configured feedback gate.

    Args:
        gate: Session feedback gate configuration.
        feedback_state: Candidate feedback summary.

    Returns:
        Gate evaluation result.
    """
    if gate is FeedbackGate.NONE:
        return FeedbackGateResult(
            gate=gate,
            satisfied=True,
            missing_sources=(),
            message="Feedback gate disabled.",
        )

    missing: list[FeedbackSource] = []
    if gate is FeedbackGate.ONE_REQUIRED:
        satisfied = feedback_state.has_human_feedback or feedback_state.has_peer_feedback
        if not satisfied:
            missing = [FeedbackSource.HUMAN, FeedbackSource.PEER]
    else:
        if not feedback_state.has_human_feedback:
            missing.append(FeedbackSource.HUMAN)
        if not feedback_state.has_peer_feedback:
            missing.append(FeedbackSource.PEER)
        satisfied = not missing

    if satisfied:
        message = (
            f"Feedback gate {gate.value} satisfied for candidate "
            f"{feedback_state.candidate_id}."
        )
    else:
        missing_sources = ", ".join(source.value for source in missing)
        message = (
            f"Feedback gate {gate.value} not satisfied for candidate "
            f"{feedback_state.candidate_id}; missing {missing_sources}."
        )
    return FeedbackGateResult(
        gate=gate,
        satisfied=satisfied,
        missing_sources=tuple(missing),
        message=message,
    )


def _clip_comment(comment: str, limit: int = 96) -> str:
    """
    Truncate long feedback comments for compact iteration summaries.

    Args:
        comment: Source comment text.
        limit: Maximum output width.

    Returns:
        Truncated summary-safe comment text.
    """
    stripped = comment.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."
