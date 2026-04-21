"""
Summary: Pydantic schema for human and peer feedback tied to reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import datetime, timezone
from rewardlab.utils.compat import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["FeedbackEntry", "FeedbackSourceType"]


class FeedbackSourceType(StrEnum):
    """Supported feedback sources."""

    HUMAN = "human"
    PEER = "peer"


class FeedbackEntry(BaseModel):
    """Validated feedback attached to a candidate iteration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    feedback_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_type: FeedbackSourceType
    score: float | None = None
    comment: str = Field(min_length=1)
    artifact_ref: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("feedback_id", "candidate_id", "comment")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        """Reject required text fields that become empty after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("artifact_ref")
    @classmethod
    def reject_blank_artifact_ref(cls, value: str | None) -> str | None:
        """Reject blank artifact references while preserving null values."""

        if value is not None and not value:
            raise ValueError("artifact_ref must not be blank when provided")
        return value


