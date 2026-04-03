"""
Summary: End-to-end CLI smoke for actual Gymnasium execution when approved dependencies exist.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rewardlab.cli.app import app


@pytest.mark.real_gymnasium
def test_cli_actual_gymnasium_smoke(monkeypatch, workspace_tmp_path: Path) -> None:
    """The CLI should complete one actual Gymnasium step when the runtime is available."""

    data_dir = workspace_tmp_path / ".rewardlab"
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("REWARDLAB_DB_PATH", str(data_dir / "metadata.sqlite3"))
    monkeypatch.setenv("REWARDLAB_EVENT_LOG_DIR", str(data_dir / "events"))
    monkeypatch.setenv("REWARDLAB_CHECKPOINT_DIR", str(data_dir / "checkpoints"))
    monkeypatch.setenv("REWARDLAB_REPORT_DIR", str(data_dir / "reports"))
    monkeypatch.setenv("REWARDLAB_EXECUTION_MODE", "actual_backend")

    objective_file = Path("tools/fixtures/objectives/cartpole.txt")
    baseline_reward_file = Path("tools/fixtures/rewards/cartpole_baseline.py")
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
            "CartPole-v1",
            "--environment-backend",
            "gymnasium",
            "--no-improve-limit",
            "2",
            "--max-iterations",
            "2",
            "--feedback-gate",
            "none",
            "--json",
        ],
    )
    assert start_result.exit_code == 0, start_result.stdout
    session_id = json.loads(start_result.stdout)["session_id"]

    step_result = runner.invoke(
        app,
        ["session", "step", "--session-id", session_id, "--json"],
    )
    assert step_result.exit_code == 0, step_result.stdout

    stop_result = runner.invoke(
        app,
        ["session", "stop", "--session-id", session_id, "--json"],
    )
    assert stop_result.exit_code == 0, stop_result.stdout
    stop_payload = json.loads(stop_result.stdout)
    report_payload = json.loads(Path(stop_payload["report_path"]).read_text(encoding="utf-8"))

    assert any(
        "actual_backend" in item["performance_summary"]
        for item in report_payload["iterations"]
    )
