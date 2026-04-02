"""
Summary: Pydantic schema for reward candidates used during iterative refinement.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["RewardCandidate"]


class RewardCandidate(BaseModel):
    """Validated representation of a reward-function candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    parent_candidate_id: str | None = Field(default=None, min_length=1)
    iteration_index: int = Field(ge=0)
    reward_definition: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    aggregate_score: float | None = None
    selected_final: bool = False
    minor_robustness_risk_accepted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("candidate_id", "session_id", "reward_definition", "change_summary")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        """Reject required text fields that become empty after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("parent_candidate_id")
    @classmethod
    def reject_blank_parent_candidate_id(cls, value: str | None) -> str | None:
        """Reject blank parent identifiers while preserving null parents."""

        if value is not None and not value:
            raise ValueError("parent_candidate_id must not be blank when provided")
        return value

    @field_validator("aggregate_score")
    @classmethod
    def allow_numeric_scores(cls, value: float | None) -> float | None:
        """Keep aggregate scores as explicit numeric values when present."""

        return value
