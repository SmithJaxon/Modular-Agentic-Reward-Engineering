"""
Summary: Unit tests for autonomous experiment spec validation and loading.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.schemas.agent_experiment import AgentExperimentSpec


def test_load_experiment_spec_from_fixture_yaml() -> None:
    """Balanced Humanoid fixture should parse into a valid experiment spec."""

    spec = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_humanoid_balanced.yaml")
    )

    assert isinstance(spec, AgentExperimentSpec)
    assert spec.environment.id == "Humanoid-v4"
    assert spec.budgets.api.max_total_tokens == 250000
    assert "run_experiment" in spec.tool_policy.allowed_tools
    assert "estimate_cost_and_risk" in spec.tool_policy.allowed_tools
    assert "compare_candidates" in spec.tool_policy.allowed_tools


def test_spec_rejects_missing_required_tool_entries() -> None:
    """Tool allowlist must include core autonomous control tools."""

    payload = {
        "version": 1,
        "experiment_name": "invalid",
        "objective": "test",
        "environment": {"backend": "gymnasium", "id": "CartPole-v1"},
        "baseline_reward": {
            "mode": "file",
            "path": "baseline.py",
            "entrypoint_name": "compute_reward",
        },
        "models": {
            "controller": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 200,
            },
            "reward_designer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 200,
            },
            "analyzer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 200,
            },
        },
        "budgets": {
            "api": {
                "max_total_tokens": 1000,
                "max_total_usd": 1.0,
                "max_completion_tokens_per_call": 200,
            },
            "time": {"max_wall_clock_minutes": 5},
            "compute": {
                "max_experiments": 2,
                "max_total_train_timesteps": 0,
                "max_parallel_experiments": 1,
            },
        },
        "governance": {
            "stopping": {
                "max_iterations": 2,
                "plateau_window": 2,
                "min_relative_improvement": 0.01,
                "max_no_improve_streak": 1,
                "max_failed_actions": 1,
            },
            "human_feedback": {"allow": False, "feedback_gate": "none", "max_requests": 0},
        },
        "tool_policy": {
            "allowed_tools": ["run_experiment"],
            "default_timeout_seconds": 60,
            "max_retries_per_tool": 1,
        },
        "execution": {"rollout": {"max_episode_steps": 10}},
        "outputs": {
            "runtime_dir": ".rewardlab",
            "report_detail": "summary",
            "save_decision_trace": True,
        },
    }

    with pytest.raises(ValueError, match="missing required entries"):
        AgentExperimentSpec.model_validate(payload)


def test_spec_rejects_required_mcp_mode_without_servers() -> None:
    """Required MCP mode must provide at least one MCP server configuration."""

    payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml")
    ).model_dump(mode="python")
    payload["tool_policy"]["mcp_execution_mode"] = "required"
    payload["tool_policy"]["mcp_servers"] = []

    with pytest.raises(ValueError, match="mcp_servers must include at least one entry"):
        AgentExperimentSpec.model_validate(payload)
