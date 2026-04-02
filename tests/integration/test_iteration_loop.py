"""
Summary: Integration tests for iterative evaluate-reflect-revise progression and ranking.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.orchestrator.session_service import SessionService
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig


@pytest.mark.integration
def test_iteration_loop_advances_candidates_and_best_selection() -> None:
    """
    Verify the orchestration loop creates candidates, reflections, and best selection pointers.
    """
    root = Path(".tmp-integration-loop") / uuid4().hex
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
            feedback_gate=FeedbackGate.NONE,
        ),
        baseline_reward_definition="reward = 1.0",
    )

    step_one = service.step_session(session["session_id"])
    assert step_one["iteration_index"] == 0
    assert step_one["candidate_id"]
    assert step_one["best_candidate_id"]

    step_two = service.step_session(session["session_id"])
    assert step_two["iteration_index"] == 1

    latest = service._require_session(session["session_id"])  # noqa: SLF001 - scoped integration check.
    assert latest["best_candidate_id"]
    candidates = service._repository.list_candidates(session["session_id"])  # noqa: SLF001
    assert len(candidates) >= 2
    shutil.rmtree(root, ignore_errors=True)
