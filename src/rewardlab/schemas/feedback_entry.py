"""
Summary: Feedback entry schema for persisted human and peer review records.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackSource(StrEnum):
    """
    Enumerate supported reviewer source channels.
    """

    HUMAN = "human"
    PEER = "peer"


class FeedbackEntry(BaseModel):
    """
    Define one persisted feedback record attached to a candidate.
    """

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_type: FeedbackSource
    score: float | None = None
    comment: str = Field(min_length=1)
    artifact_ref: str | None = None
    created_at: str = Field(min_length=1)

    @field_validator("comment")
    @classmethod
    def _strip_comment(cls, value: str) -> str:
        """
        Normalize and validate reviewer comment text.

        Args:
            value: Source comment text.

        Returns:
            Stripped comment text.

        Raises:
            ValueError: When comment is empty after trimming.
        """
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("comment cannot be blank")
        return trimmed

    @field_validator("artifact_ref")
    @classmethod
    def _strip_artifact_ref(cls, value: str | None) -> str | None:
        """
        Normalize optional artifact references.

        Args:
            value: Optional source artifact reference.

        Returns:
            Stripped artifact reference or None.

        Raises:
            ValueError: When the provided artifact reference is blank.
        """
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("artifact_ref cannot be blank")
        return trimmed
