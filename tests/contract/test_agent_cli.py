"""
Summary: Contract tests for agent run, status, events, and report commands.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from rewardlab.cli.app import app
from rewardlab.experiments.backends.base import ExperimentOutput

runner = CliRunner()


def _write_spec(path: Path) -> None:
    """
    Write a minimal JSON run spec that uses the scaffold budget tool.
    """
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_name": "agent-contract-smoke",
                "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 7},
                "objective": {
                    "text_file": "tools/fixtures/objectives/cartpole.txt",
                    "baseline_reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
                },
                "agent": {
                    "primary_model": "gpt-5.4-mini",
                    "reasoning_effort": "medium",
                    "fallback_model": "gpt-4o-mini",
                },
                "tools": {
                    "enabled": ["budget_snapshot"],
                    "max_parallel_workers": 1,
                    "per_call_timeout_seconds": 120,
                },
                "decision": {"max_turns": 3},
                "budgets": {
                    "hard": {
                        "max_wall_clock_minutes": 5,
                        "max_training_timesteps": 1000,
                        "max_evaluation_episodes": 10,
                        "max_api_input_tokens": 1000,
                        "max_api_output_tokens": 1000,
                        "max_api_usd": 1.0,
                        "max_calls_per_model": {"gpt-5.4-mini": 5},
                    },
                    "soft": {
                        "target_env_return": 100.0,
                        "plateau_window_turns": 2,
                        "min_delta_return": 1.0,
                        "min_gain_per_1k_usd": 0.1,
                        "risk_ceiling": "high",
                    },
                },
                "training_defaults": {
                    "ppo_num_envs": 1,
                    "ppo_n_steps": 128,
                    "ppo_batch_size": 128,
                    "ppo_learning_rate": 0.0003,
                },
                "reporting": {
                    "save_decision_trace": True,
                    "save_tool_trace": True,
                    "save_budget_ledger": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_experiment_spec(path: Path) -> None:
    """
    Write a run spec that routes through `run_experiment`.
    """
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_name": "agent-contract-experiment",
                "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 7},
                "objective": {
                    "text_file": "tools/fixtures/objectives/cartpole.txt",
                    "baseline_reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
                },
                "agent": {
                    "primary_model": "gpt-5.4-mini",
                    "reasoning_effort": "medium",
                    "fallback_model": "gpt-4o-mini",
                },
                "tools": {
                    "enabled": ["run_experiment"],
                    "max_parallel_workers": 1,
                    "per_call_timeout_seconds": 120,
                },
                "decision": {"max_turns": 3},
                "budgets": {
                    "hard": {
                        "max_wall_clock_minutes": 5,
                        "max_training_timesteps": 1000,
                        "max_evaluation_episodes": 10,
                        "max_api_input_tokens": 1000,
                        "max_api_output_tokens": 1000,
                        "max_api_usd": 1.0,
                        "max_calls_per_model": {"gpt-5.4-mini": 5},
                    },
                    "soft": {
                        "target_env_return": None,
                        "plateau_window_turns": 3,
                        "min_delta_return": 10.0,
                        "min_gain_per_1k_usd": 0.1,
                        "risk_ceiling": "high",
                    },
                },
                "training_defaults": {
                    "execution_mode": "deterministic",
                    "ppo_num_envs": 1,
                    "ppo_total_timesteps": 128,
                    "ppo_n_steps": 64,
                    "ppo_batch_size": 64,
                    "ppo_learning_rate": 0.0003,
                    "evaluation_episodes": 1,
                    "reflection_episodes": 0,
                    "reflection_interval_steps": 64,
                    "llm_provider": "none",
                    "llm_model": "gpt-5.4-mini",
                },
                "reporting": {
                    "save_decision_trace": True,
                    "save_tool_trace": True,
                    "save_budget_ledger": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class _FakeAdapter:
    """
    Return deterministic experiment outputs for CLI contract testing.
    """

    def run_performance(self, payload: object) -> ExperimentOutput:
        """
        Return fixed performance metrics for contract testing.
        """
        _ = payload
        return ExperimentOutput(
            score=42.0,
            metrics={"score": 42.0, "total_timesteps": 100, "evaluation_episodes_consumed": 1},
            summary="contract performance",
            artifact_refs=("contract/perf.json",),
        )

    def run_reflection(self, payload: object) -> ExperimentOutput:
        """
        Return fixed reflection summary for contract testing.
        """
        _ = payload
        return ExperimentOutput(
            score=42.0,
            metrics={"score": 42.0},
            summary="contract reflection",
            artifact_refs=("contract/refl.txt",),
        )


@pytest.mark.contract
def test_agent_run_status_events_report_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify CLI contract fields for run, status, events, and report workflow.
    """
    root = Path(".tmp-agent-cli") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    spec_file = root / "spec.json"
    _write_spec(spec_file)
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(root / "data"))

    run_result = runner.invoke(
        app,
        ["agent", "run", "--spec-file", str(spec_file), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert {"run_id", "status", "stop_reason", "turn_count", "report_path"} <= set(run_payload)
    assert run_payload["status"] == "completed"

    status_result = runner.invoke(
        app,
        ["agent", "status", "--run-id", run_payload["run_id"], "--json"],
    )
    assert status_result.exit_code == 0, status_result.stdout
    status_payload = json.loads(status_result.stdout)
    assert status_payload["run_id"] == run_payload["run_id"]
    assert status_payload["status"] == "completed"

    events_result = runner.invoke(
        app,
        ["agent", "events", "--run-id", run_payload["run_id"], "--limit", "10", "--json"],
    )
    assert events_result.exit_code == 0, events_result.stdout
    events_payload = json.loads(events_result.stdout)
    assert events_payload["run_id"] == run_payload["run_id"]
    assert events_payload["event_count"] >= 1

    report_result = runner.invoke(
        app,
        ["agent", "report", "--run-id", run_payload["run_id"], "--json"],
    )
    assert report_result.exit_code == 0, report_result.stdout
    report_payload = json.loads(report_result.stdout)
    assert report_payload["run_id"] == run_payload["run_id"]
    assert report_payload["status"] == "completed"


@pytest.mark.contract
def test_agent_run_with_run_experiment_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify agent run can execute a brokered `run_experiment` tool request.
    """
    root = Path(".tmp-agent-cli") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    spec_file = root / "spec_experiment.json"
    _write_experiment_spec(spec_file)
    monkeypatch.setenv("REWARDLAB_DATA_DIR", str(root / "data"))
    monkeypatch.setattr(
        "rewardlab.tools.executors.resolve_backend_adapter",
        lambda _: _FakeAdapter(),
    )

    run_result = runner.invoke(
        app,
        ["agent", "run", "--spec-file", str(spec_file), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["status"] == "completed"

    report_result = runner.invoke(
        app,
        ["agent", "report", "--run-id", run_payload["run_id"], "--json"],
    )
    assert report_result.exit_code == 0, report_result.stdout
    report_payload = json.loads(report_result.stdout)
    assert report_payload["tool_result_count"] >= 1
    assert report_payload["best_score"] == 42.0
    first_tool_result = report_payload["tool_results"][0]
    assert first_tool_result["tool_name"] == "run_experiment"
    assert first_tool_result["status"] == "completed"

    events_result = runner.invoke(
        app,
        ["agent", "events", "--run-id", run_payload["run_id"], "--limit", "50", "--json"],
    )
    assert events_result.exit_code == 0, events_result.stdout
    events_payload = json.loads(events_result.stdout)
    event_types = [row["event_type"] for row in events_payload["events"]]
    assert "worker.task_started" in event_types
    assert "worker.task_completed" in event_types
