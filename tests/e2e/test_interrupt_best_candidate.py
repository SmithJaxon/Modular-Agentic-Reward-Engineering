"""
Summary: End-to-end test for interruption-safe best-candidate export.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


def build_service(tmp_path: Path) -> SessionService:
    """Create a fully local session service for end-to-end interruption testing."""

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
    """Create deterministic input files for end-to-end session testing."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Keep the agent upright, centered, and smooth over time.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_interrupt_exports_best_candidate_report(tmp_path: Path) -> None:
    """Stopping a running session should export a report with the best candidate."""

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
        session_id="session-stop",
    )

    service.step_session(started.session_id)
    service.step_session(started.session_id)
    stopped = service.stop_session(started.session_id)

    assert stopped.best_candidate_id is not None
    assert stopped.stop_reason.value == "user_interrupt"
    assert stopped.report_path.exists()

    report_payload = json.loads(stopped.report_path.read_text(encoding="utf-8"))
    assert report_payload["session_id"] == started.session_id
    assert report_payload["best_candidate"]["candidate_id"] == stopped.best_candidate_id
    assert len(report_payload["iterations"]) >= 2
