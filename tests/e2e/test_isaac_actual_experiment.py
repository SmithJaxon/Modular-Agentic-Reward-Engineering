"""
Summary: End-to-end CLI smoke for actual Isaac execution when approved dependencies exist.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rewardlab.cli.app import app


@pytest.mark.real_isaacgym
def test_cli_actual_isaacgym_smoke(monkeypatch, workspace_tmp_path: Path) -> None:
    """The CLI should complete one actual Isaac step when the runtime is available."""

    environment_id = os.environ.get("REWARDLAB_TEST_ISAAC_ENV_ID")
    environment_factory = os.environ.get("REWARDLAB_ISAAC_ENV_FACTORY")
    if not environment_id or not environment_factory:
        pytest.skip(
            "set REWARDLAB_TEST_ISAAC_ENV_ID and REWARDLAB_ISAAC_ENV_FACTORY for real Isaac smoke"
        )

    data_dir = workspace_tmp_path / ".rewardlab"
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("REWARDLAB_DB_PATH", str(data_dir / "metadata.sqlite3"))
    monkeypatch.setenv("REWARDLAB_EVENT_LOG_DIR", str(data_dir / "events"))
    monkeypatch.setenv("REWARDLAB_CHECKPOINT_DIR", str(data_dir / "checkpoints"))
    monkeypatch.setenv("REWARDLAB_REPORT_DIR", str(data_dir / "reports"))
    monkeypatch.setenv("REWARDLAB_EXECUTION_MODE", "actual_backend")
    monkeypatch.setenv("REWARDLAB_ISAAC_ENV_FACTORY", environment_factory)

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
            environment_id,
            "--environment-backend",
            "isaacgym",
            "--no-improve-limit",
            "1",
            "--max-iterations",
            "1",
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
