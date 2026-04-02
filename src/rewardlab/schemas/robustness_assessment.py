"""
Summary: Robustness assessment schema and risk summarization helpers for reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(StrEnum):
    """
    Enumerate normalized robustness risk levels for candidate assessments.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RobustnessAssessment(BaseModel):
    """
    Summarize robustness degradation and reward-hacking risk for one candidate.
    """

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    variant_count: int = Field(ge=1)
    degradation_ratio: float = Field(ge=0.0)
    risk_level: RiskLevel
    risk_notes: str = Field(min_length=1)
    created_at: str = Field(min_length=1)

    @property
    def is_risk_prone(self) -> bool:
        """
        Report whether the candidate exhibits non-low robustness risk.
        """
        return self.risk_level is not RiskLevel.LOW

    def summary_line(self) -> str:
        """
        Build a concise single-line summary suitable for reports and logs.
        """
        return (
            f"{self.risk_level.value} risk across {self.variant_count} variants "
            f"(degradation={self.degradation_ratio:.3f})"
        )
