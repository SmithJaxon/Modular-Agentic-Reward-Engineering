"""
Summary: Contract tests for CLI session start, step, and stop commands.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from rewardlab.cli.app import app

runner = CliRunner()


def _setup_runtime_files(base: Path) -> tuple[Path, Path]:
    """
    Create objective and baseline files required by session start command.
    """
    objective = base / "objective.txt"
    baseline = base / "baseline.py"
    objective.write_text("maximize cartpole stability", encoding="utf-8")
    baseline.write_text("reward = 1.0", encoding="utf-8")
    return objective, baseline


@pytest.mark.contract
def test_session_start_step_stop_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify CLI contract fields for start, step, and stop workflow.
    """
    root = Path(".tmp-cli-contract") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    objective_file, baseline_file = _setup_runtime_files(root)
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(root / "data"))

    start_result = runner.invoke(
        app,
        [
            "session",
            "start",
            "--objective-file",
            str(objective_file),
            "--baseline-reward-file",
            str(baseline_file),
            "--environment-id",
            "cartpole-v1",
            "--environment-backend",
            "gymnasium",
            "--no-improve-limit",
            "3",
            "--max-iterations",
            "10",
            "--feedback-gate",
            "none",
            "--json",
        ],
    )
    assert start_result.exit_code == 0, start_result.stdout
    start_payload = json.loads(start_result.stdout)
    assert {"session_id", "status", "created_at"} <= set(start_payload)
    assert start_payload["status"] == "running"

    step_result = runner.invoke(
        app,
        ["session", "step", "--session-id", start_payload["session_id"], "--json"],
    )
    assert step_result.exit_code == 0, step_result.stdout
    step_payload = json.loads(step_result.stdout)
    assert {"session_id", "iteration_index", "candidate_id", "status", "best_candidate_id"} <= set(
        step_payload
    )
    assert step_payload["session_id"] == start_payload["session_id"]

    stop_result = runner.invoke(
        app,
        ["session", "stop", "--session-id", start_payload["session_id"], "--json"],
    )
    assert stop_result.exit_code == 0, stop_result.stdout
    stop_payload = json.loads(stop_result.stdout)
    assert stop_payload["stop_reason"] == "user_interrupt"
    assert Path(stop_payload["report_path"]).exists()
