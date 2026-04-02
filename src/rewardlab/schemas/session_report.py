"""
Summary: Session report schemas for exported orchestration results and evidence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rewardlab.schemas.session_config import EnvironmentBackend


class SessionStatus(StrEnum):
    """
    Enumerate exported terminal and resumable session states.
    """

    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(StrEnum):
    """
    Enumerate reasons for terminal or paused session exit.
    """

    USER_INTERRUPT = "user_interrupt"
    CONVERGENCE = "convergence"
    ITERATION_CAP = "iteration_cap"
    API_FAILURE_PAUSE = "api_failure_pause"
    ERROR = "error"


class RiskLevel(StrEnum):
    """
    Enumerate normalized robustness risk levels for iteration reports.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BestCandidateReport(BaseModel):
    """
    Capture summary fields for the selected best candidate.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    aggregate_score: float
    selection_summary: str = Field(min_length=1)
    minor_robustness_risk_accepted: bool = False


class IterationReport(BaseModel):
    """
    Capture one iteration summary for report exports.
    """

    model_config = ConfigDict(extra="forbid")

    iteration_index: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)
    performance_summary: str = Field(min_length=1)
    risk_level: RiskLevel
    feedback_count: int = Field(default=0, ge=0)


class SessionReport(BaseModel):
    """
    Define the exported report envelope for session state and candidates.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    status: SessionStatus
    stop_reason: StopReason
    environment_backend: EnvironmentBackend
    best_candidate: BestCandidateReport
    iterations: list[IterationReport] = Field(min_length=1)
