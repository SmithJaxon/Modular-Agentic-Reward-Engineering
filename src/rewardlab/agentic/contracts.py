"""
Summary: Shared action/result contracts for autonomous controller and tool broker.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rewardlab.schemas.agent_experiment import ActionType


class ControllerAction(BaseModel):
    """Action emitted by the controller agent for the next loop step."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: ActionType
    rationale: str = Field(min_length=1)
    expected_value: float | None = None
    expected_cost: float | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rationale")
    @classmethod
    def reject_blank_rationale(cls, value: str) -> str:
        """Reject blank rationale text."""

        if not value:
            raise ValueError("rationale must not be blank")
        return value


class ToolResult(BaseModel):
    """Normalized result returned by a worker tool execution."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    consumed_tokens: int = 0
    consumed_usd: float = 0.0

    @field_validator("status", "summary")
    @classmethod
    def reject_blank_required(cls, value: str) -> str:
        """Reject blank required text fields."""

        if not value:
            raise ValueError("value must not be blank")
        return value
