"""
Summary: Integration test for conflicting human and peer feedback handling.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


def build_service(tmp_path: Path) -> SessionService:
    """Create a local session service for feedback conflict tests."""

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
    """Create deterministic input files for conflicting feedback tests."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Reward smooth balance with centered, stable movement.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_conflicting_feedback_is_reflected_in_report_summary(tmp_path: Path) -> None:
    """Conflicting human and peer feedback should be called out in the final report."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.ONE_REQUIRED,
        session_id="session-feedback-conflict",
    )

    step = service.step_session(started.session_id)
    service.submit_human_feedback(
        session_id=started.session_id,
        candidate_id=step.candidate_id,
        comment="Observed unstable behavior near termination.",
        score=0.1,
    )
    service.request_peer_feedback(
        session_id=started.session_id,
        candidate_id=step.candidate_id,
    )

    report = service.stop_session(started.session_id)
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))

    assert "conflicting feedback" in payload["best_candidate"]["selection_summary"].lower()
    matching_iterations = [
        item
        for item in payload["iterations"]
        if item["candidate_id"] == step.candidate_id
    ]
    assert matching_iterations
    assert matching_iterations[0]["feedback_count"] == 2
