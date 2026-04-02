"""
Summary: End-to-end interruption test ensuring best-candidate export and report persistence.
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


@pytest.mark.e2e
def test_interrupt_exports_best_candidate_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify user interruption returns best candidate and writes a report artifact.
    """
    root = Path(".tmp-e2e-stop") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    objective_file = root / "objective.txt"
    baseline_file = root / "baseline.py"
    objective_file.write_text("maximize stability", encoding="utf-8")
    baseline_file.write_text("reward = 1.0", encoding="utf-8")
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
            "8",
            "--feedback-gate",
            "none",
            "--json",
        ],
    )
    assert start_result.exit_code == 0, start_result.stdout
    session_id = json.loads(start_result.stdout)["session_id"]

    for _ in range(2):
        step_result = runner.invoke(app, ["session", "step", "--session-id", session_id, "--json"])
        assert step_result.exit_code == 0, step_result.stdout

    stop_result = runner.invoke(app, ["session", "stop", "--session-id", session_id, "--json"])
    assert stop_result.exit_code == 0, stop_result.stdout
    stop_payload = json.loads(stop_result.stdout)

    report_path = Path(stop_payload["report_path"])
    assert report_path.exists()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["session_id"] == session_id
    assert report_data["best_candidate"]["candidate_id"] == stop_payload["best_candidate_id"]
    shutil.rmtree(root, ignore_errors=True)
