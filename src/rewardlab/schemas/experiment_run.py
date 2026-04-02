"""
Summary: Experiment run schema models for performance, reflection, and robustness executions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentRunType(StrEnum):
    """
    Enumerate supported experiment execution classes.
    """

    PERFORMANCE = "performance"
    REFLECTION = "reflection"
    ROBUSTNESS = "robustness"


class ExperimentRunStatus(StrEnum):
    """
    Enumerate lifecycle states for one experiment execution record.
    """

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentRun(BaseModel):
    """
    Represent one executable candidate evaluation across any experiment class.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    run_type: ExperimentRunType
    variant_label: str = Field(min_length=1)
    seed: int
    status: ExperimentRunStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    started_at: str = Field(min_length=1)
    ended_at: str | None = None

    @model_validator(mode="after")
    def _validate_run_semantics(self) -> ExperimentRun:
        """
        Enforce robustness and completion invariants from the data model.
        """
        if self.run_type is ExperimentRunType.ROBUSTNESS and self.variant_label == "default":
            raise ValueError("robustness runs must use a non-default variant label")
        if self.status is ExperimentRunStatus.COMPLETED and not self.metrics:
            raise ValueError("completed runs must include at least one metric")
        return self
