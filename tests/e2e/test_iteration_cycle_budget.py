"""
Summary: End-to-end budget test for the deterministic session iteration cycle.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


def build_service(tmp_path: Path) -> SessionService:
    """Create a fully local session service for cycle-budget validation."""

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
    """Create deterministic objective and baseline reward files for the budget test."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Reward smooth, centered balance with low oscillation.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_iteration_cycle_completes_within_local_budget(tmp_path: Path) -> None:
    """A local deterministic iteration cycle should stay well below the phase budget."""

    service = build_service(tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)

    started_at = perf_counter()
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-cycle-budget",
    )
    service.step_session(started.session_id)
    elapsed = perf_counter() - started_at

    assert elapsed < 5.0
