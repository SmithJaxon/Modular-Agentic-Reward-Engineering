"""
Summary: Foundational unit tests for schemas, retry policy, and state transitions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rewardlab.orchestrator.state_machine import (
    SessionLifecycleState,
    can_transition,
    ensure_transition,
)
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.schemas.session_report import (
    BestCandidateReport,
    IterationReport,
    RiskLevel,
    SessionReport,
    SessionStatus,
    StopReason,
)
from rewardlab.utils.retry import RetryError, RetryPolicy, run_with_retry


def test_session_config_accepts_valid_payload() -> None:
    """
    Validate that session config accepts complete, valid fields.
    """
    model = SessionConfig(
        objective_text="maximize stability",
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=10,
        feedback_gate=FeedbackGate.ONE_REQUIRED,
    )
    assert model.environment_backend is EnvironmentBackend.GYMNASIUM


def test_session_config_rejects_invalid_iteration_bounds() -> None:
    """
    Validate max_iterations cannot be smaller than no_improve_limit.
    """
    with pytest.raises(ValueError):
        SessionConfig(
            objective_text="maximize stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=5,
            max_iterations=3,
            feedback_gate=FeedbackGate.ONE_REQUIRED,
        )


def test_session_report_requires_iteration_entries() -> None:
    """
    Validate report schema requires at least one iteration summary.
    """
    with pytest.raises(ValueError):
        SessionReport(
            session_id="session-1",
            status=SessionStatus.COMPLETED,
            stop_reason=StopReason.ITERATION_CAP,
            environment_backend=EnvironmentBackend.GYMNASIUM,
            best_candidate=BestCandidateReport(
                candidate_id="cand-1",
                aggregate_score=0.8,
                selection_summary="solid",
            ),
            iterations=[],
        )


def test_session_report_accepts_valid_payload() -> None:
    """
    Validate report schema accepts well-formed payloads.
    """
    report = SessionReport(
        session_id="session-1",
        status=SessionStatus.COMPLETED,
        stop_reason=StopReason.CONVERGENCE,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=BestCandidateReport(
            candidate_id="cand-1",
            aggregate_score=0.85,
            selection_summary="improved stability",
            minor_robustness_risk_accepted=False,
        ),
        iterations=[
            IterationReport(
                iteration_index=0,
                candidate_id="cand-1",
                performance_summary="baseline pass",
                risk_level=RiskLevel.LOW,
                feedback_count=1,
            )
        ],
    )
    assert report.best_candidate.candidate_id == "cand-1"


def test_retry_returns_on_eventual_success() -> None:
    """
    Validate retry wrapper returns once operation succeeds.
    """
    attempts = {"count": 0}

    def op() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    value = run_with_retry(op, policy=RetryPolicy(max_attempts=3, base_backoff_seconds=0))
    assert value == "ok"
    assert attempts["count"] == 3


def test_retry_raises_after_exhaustion() -> None:
    """
    Validate retry wrapper raises when attempts are exhausted.
    """
    with pytest.raises(RetryError):
        run_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("always fails")),
            policy=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
        )


def test_state_machine_allows_expected_transitions() -> None:
    """
    Validate expected transitions are marked valid.
    """
    assert can_transition(SessionLifecycleState.DRAFT, SessionLifecycleState.RUNNING)
    assert can_transition(SessionLifecycleState.RUNNING, SessionLifecycleState.PAUSED)


def test_state_machine_rejects_invalid_transitions() -> None:
    """
    Validate disallowed transitions raise explicit errors.
    """
    with pytest.raises(ValueError):
        ensure_transition(SessionLifecycleState.COMPLETED, SessionLifecycleState.RUNNING)


def test_session_repository_bootstraps_local_storage() -> None:
    """
    Validate repository initializes local SQLite and accepts session writes.
    """
    base_path = Path(".tmp-test-store")
    shutil.rmtree(base_path, ignore_errors=True)
    repo = SessionRepository(base_path)
    created = repo.create_session(
        SessionConfig(
            objective_text="stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=6,
            feedback_gate=FeedbackGate.NONE,
        )
    )
    stored = repo.get_session(created["session_id"])
    assert stored is not None
    assert stored["status"] == "running"
    shutil.rmtree(base_path, ignore_errors=True)
