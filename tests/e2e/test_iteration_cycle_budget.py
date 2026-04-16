"""
Summary: End-to-end cycle budget test for deterministic session start, step, and report flow.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from rewardlab.cli.app import app

runner = CliRunner()
DETERMINISTIC_CYCLE_BUDGET_SECONDS = 10.0


@pytest.mark.e2e
def test_iteration_cycle_completes_within_deterministic_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify a deterministic start-step-report cycle stays well within the local budget.
    """
    root = Path(".tmp-e2e-cycle-budget") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    objective_file = root / "objective.txt"
    baseline_file = root / "baseline.py"
    objective_file.write_text("maximize cartpole stability", encoding="utf-8")
    baseline_file.write_text("reward = 1.0", encoding="utf-8")
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(root / "data"))

    started_at = perf_counter()
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
            "8",
            "--feedback-gate",
            "none",
            "--json",
        ],
    )
    assert start_result.exit_code == 0, start_result.stdout
    session_id = json.loads(start_result.stdout)["session_id"]

    step_result = runner.invoke(app, ["session", "step", "--session-id", session_id, "--json"])
    assert step_result.exit_code == 0, step_result.stdout

    report_result = runner.invoke(
        app,
        ["session", "report", "--session-id", session_id, "--json"],
    )
    assert report_result.exit_code == 0, report_result.stdout
    report_path = Path(json.loads(report_result.stdout)["report_path"])
    elapsed_seconds = perf_counter() - started_at

    assert report_path.exists()
    assert elapsed_seconds <= DETERMINISTIC_CYCLE_BUDGET_SECONDS, (
        "deterministic start-step-report cycle exceeded budget: "
        f"{elapsed_seconds:.3f}s > {DETERMINISTIC_CYCLE_BUDGET_SECONDS:.3f}s"
    )
    shutil.rmtree(root, ignore_errors=True)
