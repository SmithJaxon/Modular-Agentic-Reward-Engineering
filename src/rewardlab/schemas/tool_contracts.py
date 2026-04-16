"""
Summary: Schemas for brokered tool requests and results in agentic runs.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResultStatus(StrEnum):
    """
    Enumerate normalized tool execution outcomes.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ToolRequest(BaseModel):
    """
    Define one broker-validated tool execution request.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    requested_at: str = Field(min_length=1)


class ToolResult(BaseModel):
    """
    Define structured output from one brokered tool execution.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    status: ToolResultStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    artifact_refs: tuple[str, ...] = ()
    model_used: str | None = None
    api_input_tokens: int = Field(default=0, ge=0)
    api_output_tokens: int = Field(default=0, ge=0)
    api_cost_usd: float = Field(default=0.0, ge=0.0)
    training_timesteps: int = Field(default=0, ge=0)
    evaluation_episodes: int = Field(default=0, ge=0)
