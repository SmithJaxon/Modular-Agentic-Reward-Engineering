"""
Summary: Integration tests for conflicting human and peer feedback handling.
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
def test_conflicting_feedback_is_captured_in_selection_summary() -> None:
    """
    Verify opposing human and peer feedback is preserved in the final recommendation rationale.
    """
    root = Path(".tmp-feedback-conflicts") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    service = SessionService(SessionRepository(root / "data"))
    session = service.start_session(
        config=SessionConfig(
            objective_text="maximize speed while remaining aligned",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=5,
            feedback_gate=FeedbackGate.BOTH_REQUIRED,
        ),
        baseline_reward_definition="reward = exploit_speed_bonus",
    )

    step = service.step_session(session["session_id"])
    service.submit_human_feedback(
        session_id=session["session_id"],
        candidate_id=step["candidate_id"],
        comment="Human reviewer likes the quick recovery despite some rough motions.",
        score=0.35,
    )
    service.request_peer_feedback(
        session_id=session["session_id"],
        candidate_id=step["candidate_id"],
    )

    latest = service._require_session(session["session_id"])  # noqa: SLF001
    candidates = service._repository.list_candidates(session["session_id"])  # noqa: SLF001
    report = build_report_payload(latest, candidates)

    assert report.best_candidate.candidate_id == step["candidate_id"]
    assert "conflicting human and peer feedback" in report.best_candidate.selection_summary.lower()
    assert report.iterations[0].feedback_count == 2
    assert "conflicting feedback detected" in report.iterations[0].performance_summary.lower()
    shutil.rmtree(root, ignore_errors=True)
