"""
Summary: Integration test for the iterative evaluate-reflect-revise session loop.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


def build_service(tmp_path: Path) -> SessionService:
    """Create a fully local session service instance for integration tests."""

    paths = ServicePaths(
        data_dir=tmp_path / ".rewardlab",
        database_path=tmp_path / ".rewardlab" / "metadata.sqlite3",
        event_log_dir=tmp_path / ".rewardlab" / "events",
        checkpoint_dir=tmp_path / ".rewardlab" / "checkpoints",
        report_dir=tmp_path / ".rewardlab" / "reports",
    )
    service = SessionService(paths=paths)
    service.initialize()
    return service


def create_input_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create deterministic objective and baseline reward fixtures for a session."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Reward steady balance, centered cart motion, and low oscillation.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_iteration_loop_creates_candidates_reflections_and_events(tmp_path: Path) -> None:
    """A session should advance multiple deterministic iterations and retain evidence."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-loop",
    )

    step_one = service.step_session(started.session_id)
    step_two = service.step_session(started.session_id)

    candidates = service.list_candidates(started.session_id)
    reflections = service.list_reflections(started.session_id)
    events = service.read_events(started.session_id)

    assert step_one.iteration_index == 1
    assert step_two.iteration_index == 2
    assert len(candidates) == 3
    assert len(reflections) == 2
    assert len(events) >= 5
    assert candidates[-1].iteration_index == 2
    assert started.session_id == step_two.session_id
    assert step_two.best_candidate_id is not None


def test_iteration_loop_persists_reflections_for_embedded_reflection_marker(
    tmp_path: Path,
) -> None:
    """Session IDs containing `-reflection-` should still retain their reflections."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-reflection-edge",
    )

    service.step_session(started.session_id)
    reflections = service.list_reflections(started.session_id)

    assert len(reflections) == 1
    assert reflections[0].reflection_id.startswith("session-reflection-edge-reflection-")


def test_start_session_reuses_matching_explicit_session_id_without_new_side_effects(
    tmp_path: Path,
) -> None:
    """A repeated explicit `session_id` start should behave idempotently."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    first = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-idempotent-start",
    )

    second = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-idempotent-start",
    )

    events = service.read_events(first.session_id)
    candidates = service.list_candidates(first.session_id)

    assert second.session_id == first.session_id
    assert second.created_at == first.created_at
    assert len(events) == 2
    assert len(candidates) == 1


def test_start_session_rejects_conflicting_reuse_of_explicit_session_id(
    tmp_path: Path,
) -> None:
    """Reusing a session id with different start inputs should fail clearly."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-idempotent-conflict",
    )

    updated_objective_file = tmp_path / "objective-updated.txt"
    updated_objective_file.write_text(
        "Reward fast swings regardless of centering.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different configuration"):
        service.start_session(
            objective_file=updated_objective_file,
            baseline_reward_file=baseline_reward_file,
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=5,
            feedback_gate=FeedbackGate.NONE,
            session_id="session-idempotent-conflict",
        )
