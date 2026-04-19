"""
Summary: Pydantic models for session report validation and serialization.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.utils.compat import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rewardlab.schemas.session_config import EnvironmentBackend, StopReason

__all__ = [
    "EnvironmentBackend",
    "IterationSummary",
    "ReportStatus",
    "RiskLevel",
    "SelectionCandidate",
    "SessionReport",
    "StopReason",
]


class ReportStatus(StrEnum):
    """Terminal report states emitted by the orchestrator."""

    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(StrEnum):
    """Relative robustness risk labels for iteration summaries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionCandidate(BaseModel):
    """Best-known candidate included in a session report."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    aggregate_score: float
    selection_summary: str = Field(min_length=1)
    minor_robustness_risk_accepted: bool = False


class IterationSummary(BaseModel):
    """Concise summary of a single iteration in the session report."""

    model_config = ConfigDict(extra="forbid")

    iteration_index: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)
    performance_summary: str = Field(min_length=1)
    risk_level: RiskLevel
    feedback_count: int = Field(default=0, ge=0)


class SessionReport(BaseModel):
    """Validated view of a session's terminal or intermediate report."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    status: ReportStatus
    stop_reason: StopReason
    environment_backend: EnvironmentBackend
    best_candidate: SelectionCandidate
    iterations: list[IterationSummary] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_iteration_indices(self) -> Self:
        """Ensure iteration indices are unique and the report carries useful history."""

        seen: set[int] = set()
        for iteration in self.iterations:
            if iteration.iteration_index in seen:
                msg = f"duplicate iteration_index {iteration.iteration_index}"
                raise ValueError(msg)
            seen.add(iteration.iteration_index)
        return self

    @model_validator(mode="after")
    def _validate_status_stop_reason_pairing(self) -> Self:
        """Ensure the terminal status and stop reason describe the same exit mode."""

        allowed_stop_reasons = {
            ReportStatus.PAUSED: {StopReason.API_FAILURE_PAUSE},
            ReportStatus.INTERRUPTED: {StopReason.USER_INTERRUPT},
            ReportStatus.COMPLETED: {StopReason.CONVERGENCE, StopReason.ITERATION_CAP},
            ReportStatus.FAILED: {StopReason.ERROR},
        }
        if self.stop_reason not in allowed_stop_reasons[self.status]:
            expected = ", ".join(reason.value for reason in allowed_stop_reasons[self.status])
            msg = (
                f"stop_reason {self.stop_reason.value!r} is not valid for "
                f"status {self.status.value!r}; expected one of: {expected}"
            )
            raise ValueError(msg)
        return self

