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


def _minimal_valid_payload() -> dict[str, object]:
    """Return a minimal valid spec payload for schema-only tests."""

    return {
        "version": 1,
        "experiment_name": "schema-validation",
        "objective": "Validate autonomous experiment spec schema.",
        "environment": {"backend": "gymnasium", "id": "CartPole-v1"},
        "baseline_reward": {
            "mode": "file",
            "path": "tools/fixtures/rewards/cartpole_baseline.py",
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
                "max_total_tokens": 1_000,
                "max_total_usd": 1.0,
                "max_completion_tokens_per_call": 200,
            },
            "time": {"max_wall_clock_minutes": 5},
            "compute": {
                "max_experiments": 2,
                "max_total_train_timesteps": 0,
                "max_reward_generations": 2,
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
            "allowed_tools": [
                "run_experiment",
                "propose_reward_revision",
                "summarize_run_artifacts",
                "validate_reward_program",
                "estimate_cost_and_risk",
                "compare_candidates",
                "stop_or_continue_recommendation",
            ],
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


def test_load_experiment_spec_from_fixture_yaml() -> None:
    """Balanced Humanoid fixture should parse into a valid experiment spec."""

    spec = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_humanoid_balanced.yaml")
    )

    assert isinstance(spec, AgentExperimentSpec)
    assert spec.environment.id == "Humanoid-v4"
    assert spec.budgets.api.max_total_tokens == 250000
    assert spec.budgets.compute.max_reward_generations == 24
    assert spec.agent_loop.encourage_run_all_after_each_experiment is True
    assert spec.agent_loop.samples_per_iteration == 4
    assert "run_experiment" in spec.tool_policy.allowed_tools
    assert "estimate_cost_and_risk" in spec.tool_policy.allowed_tools
    assert "compare_candidates" in spec.tool_policy.allowed_tools


def test_spec_rejects_missing_required_tool_entries() -> None:
    """Tool allowlist must include core autonomous control tools."""

    payload = _minimal_valid_payload()
    payload["tool_policy"] = {
        "allowed_tools": ["run_experiment"],
        "default_timeout_seconds": 60,
        "max_retries_per_tool": 1,
    }

    with pytest.raises(ValueError, match="missing required entries"):
        AgentExperimentSpec.model_validate(payload)


def test_spec_accepts_final_evaluation_when_enabled_with_positive_runs() -> None:
    """Final evaluation should validate when enabled with num_eval_runs >= 1."""

    payload = _minimal_valid_payload()
    payload["execution"] = {
        "rollout": {"max_episode_steps": 10},
        "final_evaluation": {"enabled": True, "num_eval_runs": 3, "seed_start": 1000},
    }

    spec = AgentExperimentSpec.model_validate(payload)

    assert spec.execution.final_evaluation.enabled is True
    assert spec.execution.final_evaluation.num_eval_runs == 3


def test_spec_rejects_final_evaluation_when_enabled_with_zero_runs() -> None:
    """Final evaluation should reject enabled configs that omit usable run count."""

    payload = _minimal_valid_payload()
    payload["execution"] = {
        "rollout": {"max_episode_steps": 10},
        "final_evaluation": {"enabled": True, "num_eval_runs": 0},
    }

    with pytest.raises(
        ValueError,
        match="execution.final_evaluation.num_eval_runs must be >= 1 when enabled",
    ):
        AgentExperimentSpec.model_validate(payload)
