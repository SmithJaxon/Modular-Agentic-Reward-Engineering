"""
Summary: Contract tests for feedback CLI commands.
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
    """Create deterministic objective and baseline reward files for feedback tests."""

    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "Reward smooth balance with centered motion and stable control.",
        encoding="utf-8",
    )
    baseline_reward_file = tmp_path / "baseline_reward.py"
    baseline_reward_file.write_text(
        "def reward(state):\n    return 1.0\n",
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_feedback_submit_human_and_request_peer_contract(monkeypatch, tmp_path: Path) -> None:
    """The feedback CLI should return the documented JSON response shapes."""

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
            "one_required",
            "--json",
        ],
    )
    start_payload = json.loads(start_result.stdout)
    step_result = runner.invoke(
        app,
        ["session", "step", "--session-id", start_payload["session_id"], "--json"],
    )
    step_payload = json.loads(step_result.stdout)

    human_result = runner.invoke(
        app,
        [
            "feedback",
            "submit-human",
            "--session-id",
            start_payload["session_id"],
            "--candidate-id",
            step_payload["candidate_id"],
            "--comment",
            "Looks stable.",
            "--score",
            "0.8",
            "--artifact-ref",
            "demo.md",
            "--json",
        ],
    )

    assert human_result.exit_code == 0, human_result.stdout
    human_payload = json.loads(human_result.stdout)
    assert set(human_payload) == {"comment", "feedback_id", "source_type"}
    assert human_payload["source_type"] == "human"

    peer_result = runner.invoke(
        app,
        [
            "feedback",
            "request-peer",
            "--session-id",
            start_payload["session_id"],
            "--candidate-id",
            step_payload["candidate_id"],
            "--json",
        ],
    )

    assert peer_result.exit_code == 0, peer_result.stdout
    peer_payload = json.loads(peer_result.stdout)
    assert set(peer_payload) == {"comment", "feedback_id", "source_type"}
    assert peer_payload["source_type"] == "peer"
