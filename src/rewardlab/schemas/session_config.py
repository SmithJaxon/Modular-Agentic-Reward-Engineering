"""
Summary: Session configuration schema and shared lifecycle enums for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import datetime
from rewardlab.utils.compat import StrEnum
from typing import Union
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = Union[str, int, float, bool]


class EnvironmentBackend(StrEnum):
    """Supported environment backend values."""

    GYMNASIUM = "gymnasium"
    ISAAC_GYM = "isaacgym"


class FeedbackGate(StrEnum):
    """Supported session feedback-gating modes."""

    NONE = "none"
    ONE_REQUIRED = "one_required"
    BOTH_REQUIRED = "both_required"


class SessionStatus(StrEnum):
    """Lifecycle states for an optimization session."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(StrEnum):
    """Supported session stop reasons."""

    USER_INTERRUPT = "user_interrupt"
    CONVERGENCE = "convergence"
    ITERATION_CAP = "iteration_cap"
    API_FAILURE_PAUSE = "api_failure_pause"
    ERROR = "error"


class SessionConfig(BaseModel):
    """Validated session configuration for an optimization run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, use_enum_values=False)

    objective_text: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    environment_backend: EnvironmentBackend
    no_improve_limit: int = Field(ge=1)
    max_iterations: int = Field(ge=1)
    feedback_gate: FeedbackGate
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("objective_text", "environment_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """Reject strings that become empty after whitespace stripping."""

        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata_keys(cls, value: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
        """Ensure metadata keys are non-empty strings."""

        for key in value:
            if not key.strip():
                raise ValueError("metadata keys must not be blank")
        return value


class SessionRecord(SessionConfig):
    """Persisted session state with lifecycle fields."""

    session_id: str = Field(min_length=1)
    status: SessionStatus = SessionStatus.DRAFT
    started_at: datetime | None = None
    ended_at: datetime | None = None
    stop_reason: StopReason | None = None
    best_candidate_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def reject_blank_session_id(cls, value: str) -> str:
        """Ensure persisted session identifiers are non-empty after stripping."""

        if not value:
            raise ValueError("session_id must not be blank")
        return value

    @model_validator(mode="after")
    def validate_lifecycle_fields(self) -> SessionRecord:
        """Enforce lifecycle invariants for persisted session records."""

        terminal_statuses = {
            SessionStatus.INTERRUPTED,
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
        }

        if self.status == SessionStatus.RUNNING and self.started_at is None:
            raise ValueError("running sessions require started_at")

        if self.status in terminal_statuses and self.ended_at is None:
            raise ValueError("terminal sessions require ended_at")

        if self.ended_at is not None and self.started_at and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be earlier than started_at")

        if self.status in terminal_statuses | {SessionStatus.PAUSED} and self.stop_reason is None:
            raise ValueError("paused and terminal sessions require stop_reason")

        if self.status == SessionStatus.DRAFT and self.stop_reason is not None:
            raise ValueError("draft sessions cannot have stop_reason")

        return self


def session_record_from_mapping(data: dict[str, Any]) -> SessionRecord:
    """Validate and construct a session record from generic mapping data."""

    return SessionRecord.model_validate(data)

