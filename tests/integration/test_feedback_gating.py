"""
Summary: Integration tests for session-level feedback gating behavior and recommendation readiness.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.orchestrator.reporting import build_report_payload
from rewardlab.orchestrator.session_service import SessionService
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig


@pytest.mark.integration
@pytest.mark.parametrize(
    ("gate", "add_peer_feedback"),
    [
        (FeedbackGate.NONE, False),
        (FeedbackGate.ONE_REQUIRED, False),
        (FeedbackGate.BOTH_REQUIRED, True),
    ],
)
def test_feedback_gate_controls_final_recommendation_candidate(
    gate: FeedbackGate,
    add_peer_feedback: bool,
) -> None:
    """
    Verify the final report only recommends candidates that satisfy the configured gate.
    """
    root = Path(".tmp-feedback-gating") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    service = SessionService(SessionRepository(root / "data"))
    session = service.start_session(
        config=SessionConfig(
            objective_text="maximize stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=5,
            feedback_gate=gate,
        ),
        baseline_reward_definition="reward = exploit_speed_bonus",
    )

    step_one = service.step_session(session["session_id"])
    human_feedback = service.submit_human_feedback(
        session_id=session["session_id"],
        candidate_id=step_one["candidate_id"],
        comment="Observed stable recovery after small disturbances.",
    )
    assert human_feedback["artifact_ref"].startswith("demo://")
    if add_peer_feedback:
        service.request_peer_feedback(
            session_id=session["session_id"],
            candidate_id=step_one["candidate_id"],
        )

    step_two = service.step_session(session["session_id"])
    latest = service._require_session(session["session_id"])  # noqa: SLF001
    candidates = service._repository.list_candidates(session["session_id"])  # noqa: SLF001
    report = build_report_payload(latest, candidates)

    assert latest["best_candidate_id"] == step_two["candidate_id"]
    if gate is FeedbackGate.NONE:
        assert report.best_candidate.candidate_id == step_two["candidate_id"]
    else:
        assert report.best_candidate.candidate_id == step_one["candidate_id"]
    assert report.stop_reason is None
    assert report.iterations[0].feedback_count >= 1
    assert "feedback:" in report.iterations[0].performance_summary
    assert "final recommendation pending" not in report.best_candidate.selection_summary.lower()
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.integration
def test_feedback_gate_marks_report_pending_until_required_channels_arrive() -> None:
    """
    Verify `both_required` reports remain pending until both feedback channels exist.
    """
    root = Path(".tmp-feedback-gating") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    service = SessionService(SessionRepository(root / "data"))
    session = service.start_session(
        config=SessionConfig(
            objective_text="maximize stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=5,
            feedback_gate=FeedbackGate.BOTH_REQUIRED,
        ),
        baseline_reward_definition="reward = stability_bonus",
    )

    step = service.step_session(session["session_id"])
    service.submit_human_feedback(
        session_id=session["session_id"],
        candidate_id=step["candidate_id"],
        comment="Human reviewer sees stable balance recovery.",
        score=0.2,
    )

    pending_session = service._require_session(session["session_id"])  # noqa: SLF001
    pending_candidates = service._repository.list_candidates(session["session_id"])  # noqa: SLF001
    pending_report = build_report_payload(pending_session, pending_candidates)
    assert pending_report.stop_reason is None
    assert "final recommendation pending" in pending_report.best_candidate.selection_summary.lower()

    service.request_peer_feedback(
        session_id=session["session_id"],
        candidate_id=step["candidate_id"],
    )
    final_session = service._require_session(session["session_id"])  # noqa: SLF001
    final_candidates = service._repository.list_candidates(session["session_id"])  # noqa: SLF001
    final_report = build_report_payload(final_session, final_candidates)
    assert (
        "final recommendation pending"
        not in final_report.best_candidate.selection_summary.lower()
    )
    shutil.rmtree(root, ignore_errors=True)
