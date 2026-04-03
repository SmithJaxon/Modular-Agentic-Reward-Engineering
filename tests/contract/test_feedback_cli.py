"""
Summary: Contract tests for CLI feedback submit-human and request-peer commands.
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
def test_feedback_submit_human_and_request_peer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify CLI contract fields for human and peer feedback commands.
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
            "both_required",
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
    candidate_id = json.loads(step_result.stdout)["candidate_id"]

    human_result = runner.invoke(
        app,
        [
            "feedback",
            "submit-human",
            "--session-id",
            session_id,
            "--candidate-id",
            candidate_id,
            "--comment",
            "Observed smoother balance recovery.",
            "--score",
            "0.25",
            "--artifact-ref",
            "artifacts/cartpole-demo.mp4",
            "--json",
        ],
    )
    assert human_result.exit_code == 0, human_result.stdout
    human_payload = json.loads(human_result.stdout)
    assert {
        "feedback_id",
        "candidate_id",
        "source_type",
        "comment",
        "score",
        "artifact_ref",
    } <= set(human_payload)
    assert human_payload["source_type"] == "human"

    peer_result = runner.invoke(
        app,
        [
            "feedback",
            "request-peer",
            "--session-id",
            session_id,
            "--candidate-id",
            candidate_id,
            "--json",
        ],
    )
    assert peer_result.exit_code == 0, peer_result.stdout
    peer_payload = json.loads(peer_result.stdout)
    assert {"feedback_id", "source_type", "comment"} <= set(peer_payload)
    assert peer_payload["source_type"] == "peer"
    shutil.rmtree(root, ignore_errors=True)
