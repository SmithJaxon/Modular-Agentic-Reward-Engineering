"""
Summary: Integration test for session-level feedback gating behavior.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


def build_service(tmp_path: Path) -> SessionService:
    """Create a fully local session service instance for feedback gating tests."""

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
    """Create deterministic input files for gating tests."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Reward stable balance, centered movement, and low oscillation.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_feedback_gate_requires_both_channels_before_final_recommendation(tmp_path: Path) -> None:
    """The final recommendation summary should reflect unmet and satisfied gates."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.BOTH_REQUIRED,
        session_id="session-feedback-gate",
    )

    step = service.step_session(started.session_id)
    gated_report = service.stop_session(started.session_id)
    gated_payload = json.loads(gated_report.report_path.read_text(encoding="utf-8"))
    assert "pending required feedback" in (
        gated_payload["best_candidate"]["selection_summary"].lower()
    )

    service.submit_human_feedback(
        session_id=started.session_id,
        candidate_id=step.candidate_id,
        comment="Looks stable from the human review pass.",
        score=0.9,
    )
    still_gated_report = service.report_session(started.session_id)
    still_gated_payload = json.loads(
        still_gated_report.report_path.read_text(encoding="utf-8")
    )
    assert "pending required feedback" in (
        still_gated_payload["best_candidate"]["selection_summary"].lower()
    )

    service.request_peer_feedback(
        session_id=started.session_id,
        candidate_id=step.candidate_id,
    )
    satisfied_report = service.report_session(started.session_id)
    satisfied_payload = json.loads(satisfied_report.report_path.read_text(encoding="utf-8"))
    assert "feedback gate satisfied" in (
        satisfied_payload["best_candidate"]["selection_summary"].lower()
    )
