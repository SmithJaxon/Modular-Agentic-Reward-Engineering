"""
Summary: Unit tests for agentic run spec loading from constrained YAML and JSON.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.agentic.spec_loader import load_run_spec


@pytest.mark.unit
def test_load_run_spec_parses_repository_yaml_profile() -> None:
    """
    Verify constrained YAML parser accepts the committed humanoid scout profile.
    """
    path = Path("configs/agentic/humanoid_scout.yaml")
    spec = load_run_spec(path)
    assert spec.run_name == "humanoid_scout"
    assert spec.environment.id == "Humanoid-v4"
    assert "run_experiment" in spec.tools.enabled


@pytest.mark.unit
def test_load_run_spec_parses_openai_planner_profile() -> None:
    """
    Verify agent profile planner-provider settings parse from YAML config.
    """
    path = Path("configs/agentic/humanoid_main_openai.yaml")
    spec = load_run_spec(path)
    assert spec.run_name == "humanoid_main_openai"
    assert spec.agent.planner_provider == "openai"
    assert spec.agent.planner_context_window >= 1


@pytest.mark.unit
def test_load_run_spec_parses_json_profile() -> None:
    """
    Verify JSON specs are accepted and validated with the same contract.
    """
    root = Path(".tmp-agentic-spec") / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    path = root / "spec.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_name": "json-smoke",
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
    spec = load_run_spec(path)
    assert spec.run_name == "json-smoke"
    assert spec.tools.enabled == ("budget_snapshot",)
