"""
Summary: Integration tests for routing experiment execution through selected backends.
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
@pytest.mark.parametrize(
    ("backend", "environment_id"),
    [
        (EnvironmentBackend.GYMNASIUM, "cartpole-v1"),
        (EnvironmentBackend.ISAACGYM, "isaac-ant-v0"),
    ],
)
def test_environment_backend_routes_iteration_through_matching_adapter(
    backend: EnvironmentBackend,
    environment_id: str,
) -> None:
    """
    Verify session stepping resolves the adapter that matches the session backend.
    """
    root = Path(".tmp-backend-selection") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    service = SessionService(SessionRepository(root / "data"))
    session = service.start_session(
        config=SessionConfig(
            objective_text="maximize stability",
            environment_id=environment_id,
            environment_backend=backend,
            no_improve_limit=3,
            max_iterations=5,
            feedback_gate=FeedbackGate.NONE,
        ),
        baseline_reward_definition="reward = shaped_progress_bonus",
    )

    step = service.step_session(session["session_id"])
    updated = service._require_session(session["session_id"])  # noqa: SLF001 - integration probe.
    assessment = updated["metadata"]["robustness_assessments"][step["candidate_id"]]

    assert step["performance_summary"].startswith(f"{backend.value} performance")
    assert assessment["variant_count"] >= 1
    assert assessment["risk_level"] in {"low", "medium", "high"}
    shutil.rmtree(root, ignore_errors=True)
