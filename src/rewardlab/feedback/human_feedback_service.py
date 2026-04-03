"""
Summary: Human feedback ingestion service for validated candidate review submissions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from typing import Any

from rewardlab.feedback.demo_artifacts import DemoArtifactTracker
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSource


class HumanFeedbackService:
    """
    Validate and persist structured human feedback submissions.
    """

    def __init__(
        self,
        repository: SessionRepository,
        artifact_tracker: DemoArtifactTracker | None = None,
    ) -> None:
        """
        Initialize human feedback dependencies.

        Args:
            repository: Session repository facade.
            artifact_tracker: Optional demonstration artifact tracker override.
        """
        self._repository = repository
        self._artifact_tracker = artifact_tracker or DemoArtifactTracker()

    def submit_feedback(
        self,
        session: dict[str, Any],
        candidate: dict[str, Any],
        comment: str,
        score: float | None = None,
        artifact_ref: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist one human feedback record for a session candidate.

        Args:
            session: Session metadata dictionary.
            candidate: Candidate metadata dictionary.
            comment: Reviewer comment text.
            score: Optional normalized score delta.
            artifact_ref: Optional demonstration artifact reference.

        Returns:
            Persisted feedback metadata dictionary.
        """
        if candidate["session_id"] != session["session_id"]:
            raise RuntimeError(
                f"candidate {candidate['candidate_id']} does not belong to session "
                f"{session['session_id']}"
            )
        resolved_artifact_ref, metadata = self._artifact_tracker.register_artifact(
            session=session,
            candidate_id=candidate["candidate_id"],
            artifact_ref=artifact_ref,
        )
        self._repository.update_session(session["session_id"], metadata=metadata)
        payload = self._repository.add_feedback(
            candidate_id=candidate["candidate_id"],
            source_type=FeedbackSource.HUMAN.value,
            comment=comment,
            score=score,
            artifact_ref=resolved_artifact_ref,
        )
        return FeedbackEntry.model_validate(payload).model_dump()
