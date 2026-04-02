"""
Summary: Session configuration schema and validation helpers for orchestrator startup.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnvironmentBackend(StrEnum):
    """
    Enumerate supported environment backends for session execution.
    """

    GYMNASIUM = "gymnasium"
    ISAACGYM = "isaacgym"


class FeedbackGate(StrEnum):
    """
    Enumerate supported feedback gating modes.
    """

    NONE = "none"
    ONE_REQUIRED = "one_required"
    BOTH_REQUIRED = "both_required"


class SessionConfig(BaseModel):
    """
    Define validated runtime configuration for a reward optimization session.
    """

    model_config = ConfigDict(extra="forbid")

    objective_text: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    environment_backend: EnvironmentBackend
    no_improve_limit: int = Field(gt=0)
    max_iterations: int = Field(gt=0)
    feedback_gate: FeedbackGate
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("objective_text", "environment_id")
    @classmethod
    def _strip_text_fields(cls, value: str) -> str:
        """
        Normalize and validate non-empty string fields.

        Args:
            value: Source text value.

        Returns:
            Stripped text value.

        Raises:
            ValueError: When value is empty after whitespace trimming.
        """
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("field cannot be blank")
        return trimmed

    @field_validator("max_iterations")
    @classmethod
    def _validate_max_iterations(cls, value: int, info: Any) -> int:
        """
        Ensure max iteration cap is not less than no-improvement threshold.

        Args:
            value: Candidate maximum iteration count.
            info: Pydantic validation context.

        Returns:
            Validated maximum iteration count.

        Raises:
            ValueError: When max_iterations is smaller than no_improve_limit.
        """
        no_improve_limit = info.data.get("no_improve_limit")
        if isinstance(no_improve_limit, int) and value < no_improve_limit:
            raise ValueError("max_iterations must be greater than or equal to no_improve_limit")
        return value
