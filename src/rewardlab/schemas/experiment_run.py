"""
Summary: Pydantic schema for experiment runs used in iterative reward evaluation.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import datetime
from rewardlab.utils.compat import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rewardlab.schemas.session_config import EnvironmentBackend

__all__ = ["ExecutionMode", "ExperimentRun", "RunStatus", "RunType"]


class RunType(StrEnum):
    """Supported experiment run categories."""

    PERFORMANCE = "performance"
    REFLECTION = "reflection"
    ROBUSTNESS = "robustness"


class RunStatus(StrEnum):
    """Supported experiment run lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionMode(StrEnum):
    """Supported execution modes for experiment runs."""

    OFFLINE_TEST = "offline_test"
    ACTUAL_BACKEND = "actual_backend"


class ExperimentRun(BaseModel):
    """Validated representation of a single candidate evaluation run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    backend: EnvironmentBackend
    environment_id: str = Field(min_length=1)
    run_type: RunType
    execution_mode: ExecutionMode = ExecutionMode.OFFLINE_TEST
    variant_label: str = Field(min_length=1)
    seed: int | None = None
    status: RunStatus = RunStatus.QUEUED
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @field_validator("run_id", "candidate_id", "environment_id", "variant_label")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        """Reject required text fields that become empty after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: list[str]) -> list[str]:
        """Ensure artifact references are non-empty strings when present."""

        for item in value:
            if not item:
                raise ValueError("artifact_refs entries must not be blank")
        return value

    @field_validator("failure_reason")
    @classmethod
    def reject_blank_failure_reason(cls, value: str | None) -> str | None:
        """Reject blank failure reasons while preserving null values."""

        if value is not None and not value:
            raise ValueError("failure_reason must not be blank when provided")
        return value

    @model_validator(mode="after")
    def validate_run_shape(self) -> ExperimentRun:
        """Enforce run-type-specific and terminal-state invariants."""

        if self.run_type == RunType.ROBUSTNESS and _is_default_variant_label(self.variant_label):
            raise ValueError("robustness runs must use a non-default variant_label")

        if self.status == RunStatus.COMPLETED and not self.metrics:
            raise ValueError("completed runs must include at least one metric entry")

        if (
            self.execution_mode == ExecutionMode.ACTUAL_BACKEND
            and self.status == RunStatus.COMPLETED
            and not self.artifact_refs
        ):
            raise ValueError("completed actual_backend runs must include artifact_refs")

        if self.status == RunStatus.FAILED and not self.failure_reason:
            raise ValueError("failed runs must include failure_reason")

        if self.status != RunStatus.FAILED and self.failure_reason is not None:
            raise ValueError("only failed runs may include failure_reason")

        if self.status in {RunStatus.COMPLETED, RunStatus.FAILED} and self.ended_at is None:
            raise ValueError("terminal runs must include ended_at")

        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("ended_at cannot be earlier than started_at")

        return self


def _is_default_variant_label(value: str) -> bool:
    """Return whether a variant label names the default baseline condition."""

    return value.strip().casefold() in {"default", "baseline", "control"}

