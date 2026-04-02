"""
Summary: Session state transition rules for orchestrator lifecycle enforcement.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from enum import StrEnum


class SessionLifecycleState(StrEnum):
    """
    Enumerate allowed lifecycle states for optimization sessions.
    """

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[SessionLifecycleState, set[SessionLifecycleState]] = {
    SessionLifecycleState.DRAFT: {SessionLifecycleState.RUNNING},
    SessionLifecycleState.RUNNING: {
        SessionLifecycleState.PAUSED,
        SessionLifecycleState.INTERRUPTED,
        SessionLifecycleState.COMPLETED,
        SessionLifecycleState.FAILED,
    },
    SessionLifecycleState.PAUSED: {SessionLifecycleState.RUNNING, SessionLifecycleState.FAILED},
    SessionLifecycleState.INTERRUPTED: set(),
    SessionLifecycleState.COMPLETED: set(),
    SessionLifecycleState.FAILED: set(),
}


def can_transition(current: SessionLifecycleState, target: SessionLifecycleState) -> bool:
    """
    Evaluate whether the requested lifecycle transition is valid.

    Args:
        current: Current session lifecycle state.
        target: Requested target state.

    Returns:
        True if transition is allowed, otherwise False.
    """
    return target in ALLOWED_TRANSITIONS[current]


def ensure_transition(current: SessionLifecycleState, target: SessionLifecycleState) -> None:
    """
    Assert that a lifecycle transition is valid.

    Args:
        current: Current state value.
        target: Target state value.

    Raises:
        ValueError: If transition is not defined in allowed transitions map.
    """
    if not can_transition(current, target):
        raise ValueError(f"invalid transition from {current.value} to {target.value}")
