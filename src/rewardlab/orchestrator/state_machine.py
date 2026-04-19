"""
Summary: Session lifecycle transition rules for RewardLab orchestration.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from rewardlab.schemas.session_config import SessionRecord, SessionStatus, StopReason

ALLOWED_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.DRAFT: {SessionStatus.RUNNING},
    SessionStatus.RUNNING: {
        SessionStatus.PAUSED,
        SessionStatus.INTERRUPTED,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
    },
    SessionStatus.PAUSED: {SessionStatus.RUNNING},
    SessionStatus.INTERRUPTED: set(),
    SessionStatus.COMPLETED: set(),
    SessionStatus.FAILED: set(),
}


@dataclass(frozen=True)
class TransitionRequest:
    """Requested state transition with optional lifecycle annotations."""

    next_status: SessionStatus
    stop_reason: StopReason | None = None
    best_candidate_id: str | None = None
    occurred_at: datetime | None = None


def can_transition(current_status: SessionStatus, next_status: SessionStatus) -> bool:
    """Return True when a lifecycle transition is allowed."""

    return next_status in ALLOWED_TRANSITIONS[current_status]


def apply_transition(record: SessionRecord, request: TransitionRequest) -> SessionRecord:
    """Apply a validated state transition to a session record."""

    if not can_transition(record.status, request.next_status):
        raise ValueError(f"invalid transition: {record.status} -> {request.next_status}")

    occurred_at = request.occurred_at or datetime.now(timezone.utc)
    updates: dict[str, object] = {
        "status": request.next_status,
        "best_candidate_id": request.best_candidate_id or record.best_candidate_id,
    }

    if request.next_status == SessionStatus.RUNNING:
        updates["started_at"] = record.started_at or occurred_at
        updates["ended_at"] = None
        updates["stop_reason"] = None
    elif request.next_status == SessionStatus.PAUSED:
        updates["stop_reason"] = request.stop_reason or StopReason.API_FAILURE_PAUSE
    else:
        updates["ended_at"] = occurred_at
        updates["stop_reason"] = request.stop_reason or _default_stop_reason(request.next_status)

    return record.model_copy(update=updates)


def _default_stop_reason(status: SessionStatus) -> StopReason:
    """Map terminal statuses to their default stop reasons."""

    if status == SessionStatus.INTERRUPTED:
        return StopReason.USER_INTERRUPT
    if status == SessionStatus.COMPLETED:
        return StopReason.CONVERGENCE
    if status == SessionStatus.FAILED:
        return StopReason.ERROR
    raise ValueError(f"status {status} does not have a default stop reason")

