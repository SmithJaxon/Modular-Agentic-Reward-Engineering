"""
Summary: Contract tests for session lifecycle CLI commands.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rewardlab.cli.app import app


def configure_runtime_env(monkeypatch, tmp_path: Path) -> None:
    """Point the CLI at a test-local RewardLab runtime directory tree."""

    data_dir = tmp_path / ".rewardlab"
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("REWARDLAB_DB_PATH", str(data_dir / "metadata.sqlite3"))
    monkeypatch.setenv("REWARDLAB_EVENT_LOG_DIR", str(data_dir / "events"))
    monkeypatch.setenv("REWARDLAB_CHECKPOINT_DIR", str(data_dir / "checkpoints"))
    monkeypatch.setenv("REWARDLAB_REPORT_DIR", str(data_dir / "reports"))


def create_input_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create deterministic objective and baseline reward files for CLI tests."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Balance the pole while keeping the cart centered and stable.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_session_start_step_and_stop_contract(monkeypatch, tmp_path: Path) -> None:
    """The session lifecycle CLI should return the documented JSON response shapes."""

    configure_runtime_env(monkeypatch, tmp_path)
    objective_file, baseline_reward_file = create_input_files(tmp_path)
    runner = CliRunner()

    start_result = runner.invoke(
        app,
        [
            "session",
            "start",
            "--objective-file",
            str(objective_file),
            "--baseline-reward-file",
            str(baseline_reward_file),
            "--environment-id",
            "cartpole-v1",
            "--environment-backend",
            "gymnasium",
            "--no-improve-limit",
            "2",
            "--max-iterations",
            "4",
            "--feedback-gate",
            "none",
            "--json",
        ],
    )

    assert start_result.exit_code == 0, start_result.stdout
    start_payload = json.loads(start_result.stdout)
    assert set(start_payload) == {"created_at", "session_id", "status"}
    assert start_payload["status"] == "running"

    session_id = start_payload["session_id"]

    step_result = runner.invoke(
        app,
        ["session", "step", "--session-id", session_id, "--json"],
    )

    assert step_result.exit_code == 0, step_result.stdout
    step_payload = json.loads(step_result.stdout)
    assert set(step_payload) == {
        "best_candidate_id",
        "candidate_id",
        "iteration_index",
        "session_id",
        "status",
    }
    assert step_payload["session_id"] == session_id
    assert step_payload["iteration_index"] == 1

    stop_result = runner.invoke(
        app,
        ["session", "stop", "--session-id", session_id, "--json"],
    )

    assert stop_result.exit_code == 0, stop_result.stdout
    stop_payload = json.loads(stop_result.stdout)
    assert set(stop_payload) == {
        "best_candidate_id",
        "report_path",
        "session_id",
        "stop_reason",
    }
    assert Path(stop_payload["report_path"]).exists()

    report_result = runner.invoke(
        app,
        ["session", "report", "--session-id", session_id, "--json"],
    )

    assert report_result.exit_code == 0, report_result.stdout
    report_payload = json.loads(report_result.stdout)
    assert set(report_payload) == {
        "best_candidate_id",
        "report_path",
        "session_id",
        "stop_reason",
    }
    assert Path(report_payload["report_path"]).exists()
