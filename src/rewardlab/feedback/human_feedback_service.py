"""
Summary: Human feedback creation and artifact bundle management for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime

from rewardlab.feedback.demo_artifacts import DemoArtifactTracker
from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSourceType


class HumanFeedbackService:
    """Create structured human feedback entries and optional local review bundles."""

    def __init__(self, artifact_tracker: DemoArtifactTracker | None = None) -> None:
        """Store the optional artifact tracker used for human review bundles."""

        self.artifact_tracker = artifact_tracker

    def submit_feedback(
        self,
        *,
        session_id: str,
        candidate_id: str,
        comment: str,
        score: float | None = None,
        artifact_ref: str | None = None,
    ) -> FeedbackEntry:
        """Create a human feedback entry and persist a lightweight artifact bundle."""

        feedback = FeedbackEntry(
            feedback_id=_feedback_id(session_id, FeedbackSourceType.HUMAN),
            candidate_id=candidate_id,
            source_type=FeedbackSourceType.HUMAN,
            score=score,
            comment=comment,
            artifact_ref=artifact_ref,
        )
        if self.artifact_tracker is None:
            return feedback

        bundle = self.artifact_tracker.write_feedback_bundle(
            feedback,
            title="Human Feedback Review",
            metadata={"session_id": session_id},
        )
        resolved_artifact_ref = artifact_ref or str(bundle.review_markdown_path)
        return feedback.model_copy(update={"artifact_ref": resolved_artifact_ref})


def _feedback_id(session_id: str, source_type: FeedbackSourceType) -> str:
    """Return a timestamp-based feedback identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{session_id}-{source_type.value}-feedback-{timestamp}"
