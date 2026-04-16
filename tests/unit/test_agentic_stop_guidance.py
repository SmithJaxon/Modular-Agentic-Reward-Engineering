"""
Summary: Unit tests for agentic stop-guidance heuristics.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.stop_guidance import StopGuidance
from rewardlab.schemas.agentic_run import AgenticRunSpec, StopDecisionTag
from rewardlab.schemas.tool_contracts import ToolResult, ToolResultStatus


def _build_spec(
    *,
    target_env_return: float | None = None,
    plateau_window_turns: int = 3,
    min_delta_return: float = 0.01,
    min_gain_per_1k_usd: float = 0.0,
    risk_ceiling: str = "high",
) -> AgenticRunSpec:
    """
    Build a compact agentic run spec for stop-guidance tests.
    """
    return AgenticRunSpec.model_validate(
        {
            "version": 1,
            "run_name": "stop-guidance-test",
            "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 7},
            "objective": {
                "text_file": "tools/fixtures/objectives/cartpole.txt",
                "baseline_reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
            },
            "agent": {
                "primary_model": "gpt-5.4-mini",
                "reasoning_effort": "high",
                "fallback_model": "gpt-4o-mini",
            },
            "tools": {
                "enabled": ["run_experiment", "run_probe_suite", "export_report"],
                "max_parallel_workers": 1,
                "per_call_timeout_seconds": 120,
            },
            "decision": {"max_turns": 20},
            "budgets": {
                "hard": {
                    "max_wall_clock_minutes": 30,
                    "max_training_timesteps": 10000,
                    "max_evaluation_episodes": 100,
                    "max_api_input_tokens": 100000,
                    "max_api_output_tokens": 100000,
                    "max_api_usd": 10.0,
                    "max_calls_per_model": {"gpt-5.4-mini": 20},
                },
                "soft": {
                    "target_env_return": target_env_return,
                    "plateau_window_turns": plateau_window_turns,
                    "min_delta_return": min_delta_return,
                    "min_gain_per_1k_usd": min_gain_per_1k_usd,
                    "risk_ceiling": risk_ceiling,
                },
            },
        }
    )


def _completed_result(
    *,
    turn_index: int,
    tool_name: str,
    output: dict[str, object],
) -> ToolResult:
    """
    Build a completed tool result row for context-store tests.
    """
    return ToolResult(
        request_id=f"toolreq-{turn_index}",
        turn_index=turn_index,
        tool_name=tool_name,
        status=ToolResultStatus.COMPLETED,
        output=output,
    )


def test_stop_guidance_returns_objective_met_when_target_reached() -> None:
    """
    Verify objective stop is emitted when best score reaches target threshold.
    """
    spec = _build_spec(target_env_return=0.90)
    context = ContextStore()
    context.record_tool_result(
        _completed_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-0", "score": 0.93},
        )
    )
    budget_state = BudgetEngine.from_spec(spec).state

    decision = StopGuidance().evaluate(spec=spec, context=context, budget_state=budget_state)

    assert decision is not None
    assert decision.tag is StopDecisionTag.OBJECTIVE_MET


def test_stop_guidance_returns_plateau_for_flat_recent_scores() -> None:
    """
    Verify plateau stop is emitted when recent score spread is below threshold.
    """
    spec = _build_spec(
        target_env_return=None,
        plateau_window_turns=3,
        min_delta_return=0.05,
    )
    context = ContextStore()
    context.record_tool_result(
        _completed_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-0", "score": 0.50},
        )
    )
    context.record_tool_result(
        _completed_result(
            turn_index=1,
            tool_name="run_experiment",
            output={"candidate_id": "cand-1", "score": 0.53},
        )
    )
    context.record_tool_result(
        _completed_result(
            turn_index=2,
            tool_name="run_experiment",
            output={"candidate_id": "cand-2", "score": 0.52},
        )
    )
    budget_state = BudgetEngine.from_spec(spec).state

    decision = StopGuidance().evaluate(spec=spec, context=context, budget_state=budget_state)

    assert decision is not None
    assert decision.tag is StopDecisionTag.PLATEAU


def test_stop_guidance_returns_risk_limit_when_probe_exceeds_ceiling() -> None:
    """
    Verify risk-limit stop is emitted when probe risk level is above ceiling.
    """
    spec = _build_spec(
        target_env_return=None,
        plateau_window_turns=5,
        min_delta_return=0.0,
        risk_ceiling="low",
    )
    context = ContextStore()
    context.record_tool_result(
        _completed_result(
            turn_index=0,
            tool_name="run_probe_suite",
            output={"candidate_id": "cand-risk", "risk_level": "high"},
        )
    )
    budget_state = BudgetEngine.from_spec(spec).state

    decision = StopGuidance().evaluate(spec=spec, context=context, budget_state=budget_state)

    assert decision is not None
    assert decision.tag is StopDecisionTag.RISK_LIMIT


def test_stop_guidance_returns_cost_inefficient_for_low_gain_per_cost() -> None:
    """
    Verify cost-inefficient stop is emitted when gain-per-USD is below minimum.
    """
    spec = _build_spec(
        target_env_return=None,
        plateau_window_turns=5,
        min_delta_return=0.0,
        min_gain_per_1k_usd=20.0,
    )
    context = ContextStore()
    context.record_tool_result(
        _completed_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-0", "score": 0.50},
        )
    )
    context.record_tool_result(
        _completed_result(
            turn_index=1,
            tool_name="run_experiment",
            output={"candidate_id": "cand-1", "score": 0.55},
        )
    )
    budget_state = BudgetEngine.from_spec(spec).state
    budget_state.usage.api_cost_usd = 5.0

    decision = StopGuidance().evaluate(spec=spec, context=context, budget_state=budget_state)

    assert decision is not None
    assert decision.tag is StopDecisionTag.COST_INEFFICIENT


def test_stop_guidance_returns_cost_inefficient_on_hard_budget_exhaustion() -> None:
    """
    Verify hard-budget exhaustion emits a cost-inefficient stop recommendation.
    """
    spec = _build_spec(target_env_return=None)
    context = ContextStore()
    budget_state = BudgetEngine.from_spec(spec).state
    budget_state.usage.training_timesteps = budget_state.max_training_timesteps

    decision = StopGuidance().evaluate(spec=spec, context=context, budget_state=budget_state)

    assert decision is not None
    assert decision.tag is StopDecisionTag.COST_INEFFICIENT
    assert "training_timesteps" in decision.reason
