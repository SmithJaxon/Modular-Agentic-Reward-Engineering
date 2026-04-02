"""
Summary: Reward candidate schema for iterative reward definition revisions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RewardCandidate(BaseModel):
    """
    Represent one candidate reward function revision in a session.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    iteration_index: int = Field(ge=0)
    reward_definition: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    aggregate_score: float
    created_at: str = Field(min_length=1)
