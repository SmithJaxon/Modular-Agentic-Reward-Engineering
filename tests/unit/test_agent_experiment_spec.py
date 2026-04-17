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


def test_spec_defaults_to_enforcing_progress_before_stop() -> None:
    """Agent loop should default to enforcing minimum progress before STOP actions."""

    spec = AgentExperimentSpec.model_validate(_minimal_valid_payload())

    assert spec.agent_loop.enforce_progress_before_stop is True


def test_spec_allows_disabling_progress_before_stop_enforcement() -> None:
    """Spec should accept explicit opt-out for progress-before-stop enforcement."""

    payload = _minimal_valid_payload()
    payload["agent_loop"] = {
        "encourage_run_all_after_each_experiment": False,
        "samples_per_iteration": 1,
        "enforce_progress_before_stop": False,
    }

    spec = AgentExperimentSpec.model_validate(payload)

    assert spec.agent_loop.enforce_progress_before_stop is False


def test_spec_accepts_ppo_parallel_env_and_device_controls() -> None:
    """Spec should accept explicit PPO parallel-env and device settings."""

    payload = _minimal_valid_payload()
    payload["execution"] = {
        "ppo": {
            "total_timesteps": 50000,
            "eval_runs": 3,
            "checkpoint_count": 10,
            "eval_episodes_per_checkpoint": 1,
            "n_envs": 8,
            "device": "cuda:0",
        }
    }

    spec = AgentExperimentSpec.model_validate(payload)

    assert spec.execution.ppo is not None
    assert spec.execution.ppo.n_envs == 8
    assert spec.execution.ppo.device == "cuda:0"


def test_spec_rejects_non_positive_ppo_n_envs() -> None:
    """PPO parallel env count must be a positive integer."""

    payload = _minimal_valid_payload()
    payload["execution"] = {
        "ppo": {
            "total_timesteps": 50000,
            "eval_runs": 3,
            "checkpoint_count": 10,
            "eval_episodes_per_checkpoint": 1,
            "n_envs": 0,
            "device": "auto",
        }
    }

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        AgentExperimentSpec.model_validate(payload)
