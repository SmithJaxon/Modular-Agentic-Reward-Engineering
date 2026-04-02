"""
Summary: Pydantic schema for robustness assessments used in reward-hacking analysis.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["RobustnessAssessment", "RiskLevel"]


class RiskLevel(StrEnum):
    """Risk classifications for candidate robustness analysis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RobustnessAssessment(BaseModel):
    """Validated summary of a candidate's robustness and reward-hacking risk."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assessment_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    variant_count: int = Field(ge=1)
    degradation_ratio: float = Field(ge=0.0)
    risk_level: RiskLevel
    risk_notes: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("assessment_id", "candidate_id", "risk_notes")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        """Reject required text fields that become empty after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value
