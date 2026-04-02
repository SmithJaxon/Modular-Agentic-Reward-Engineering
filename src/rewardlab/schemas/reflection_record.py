"""
Summary: Reflection schema for rationale and proposed changes per candidate iteration.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ReflectionRecord(BaseModel):
    """
    Represent one reflection attached to a candidate iteration.
    """

    model_config = ConfigDict(extra="forbid")

    reflection_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    proposed_changes: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str = Field(min_length=1)
