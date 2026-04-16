"""
Summary: Unit tests for primary optimizer decision policy behavior.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

import pytest

from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.llm_planner import PlannerAttemptFeedback, PlannerCallUsage
from rewardlab.agentic.primary_optimizer import PrimaryOptimizer
from rewardlab.schemas.agentic_run import (
    AgentDecision,
    AgentDecisionAction,
    AgenticRunSpec,
    StopDecisionTag,
)
from rewardlab.schemas.budget_state import BudgetState
from rewardlab.schemas.tool_contracts import ToolResult, ToolResultStatus


class _PlannerStubBase:
    """
    Provide optional planner protocol methods for local test doubles.
    """

    def last_usage(self) -> PlannerCallUsage | None:
        """
        Return no usage by default for planner stub tests.
        """
        return None

    def last_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return no planner feedback by default for planner stub tests.
        """
        return ()


def _build_spec() -> AgenticRunSpec:
    """
    Build a policy-focused run spec for primary optimizer tests.
    """
    return AgenticRunSpec.model_validate(
        {
            "version": 1,
            "run_name": "policy-test",
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
                "enabled": [
                    "run_experiment",
                    "run_probe_suite",
                    "compare_candidates",
                    "export_report",
                ],
                "max_parallel_workers": 1,
                "per_call_timeout_seconds": 120,
            },
            "decision": {
                "max_turns": 20,
                "min_candidates_before_compare": 2,
                "compare_every_new_candidates": 2,
                "require_probe_before_compare": True,
            },
            "budgets": {
                "hard": {
                    "max_wall_clock_minutes": 30,
                    "max_training_timesteps": 50000,
                    "max_evaluation_episodes": 100,
                    "max_api_input_tokens": 100000,
                    "max_api_output_tokens": 100000,
                    "max_api_usd": 10.0,
                    "max_calls_per_model": {"gpt-5.4-mini": 20},
                },
                "soft": {
                    "target_env_return": None,
                    "plateau_window_turns": 6,
                    "min_delta_return": 0.01,
                    "min_gain_per_1k_usd": 0.0,
                    "risk_ceiling": "high",
                },
            },
            "training_defaults": {
                "execution_mode": "deterministic",
                "ppo_num_envs": 1,
                "ppo_total_timesteps": 4096,
                "ppo_n_steps": 128,
                "ppo_batch_size": 128,
                "ppo_learning_rate": 0.0003,
                "evaluation_episodes": 5,
                "reflection_episodes": 2,
                "reflection_interval_steps": 1024,
                "llm_provider": "none",
                "llm_model": "gpt-5.4-mini",
            },
        }
    )


def _budget_state() -> BudgetState:
    """
    Build budget state with ample remaining capacity.
    """
    return BudgetState(
        max_wall_clock_minutes=30,
        max_training_timesteps=50000,
        max_evaluation_episodes=100,
        max_api_input_tokens=100000,
        max_api_output_tokens=100000,
        max_api_usd=10.0,
        max_calls_per_model={"gpt-5.4-mini": 20},
    )


def _tool_result(
    *,
    turn_index: int,
    tool_name: str,
    output: dict[str, object],
) -> ToolResult:
    """
    Build a completed tool result row for context seeding.
    """
    return ToolResult(
        request_id=f"toolreq-{turn_index}",
        turn_index=turn_index,
        tool_name=tool_name,
        status=ToolResultStatus.COMPLETED,
        output=output,
    )


@pytest.mark.unit
def test_primary_optimizer_compares_after_probed_candidate_threshold() -> None:
    """
    Verify compare-candidates is dispatched once threshold and probe requirements are met.
    """
    spec = _build_spec()
    context = ContextStore()
    context.record_tool_result(
        _tool_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-a", "score": 0.50},
        )
    )
    context.record_tool_result(
        _tool_result(
            turn_index=1,
            tool_name="run_probe_suite",
            output={"candidate_id": "cand-a", "risk_level": "low", "robustness_bonus": 0.05},
        )
    )
    context.record_tool_result(
        _tool_result(
            turn_index=2,
            tool_name="run_experiment",
            output={"candidate_id": "cand-b", "score": 0.56},
        )
    )
    context.record_tool_result(
        _tool_result(
            turn_index=3,
            tool_name="run_probe_suite",
            output={"candidate_id": "cand-b", "risk_level": "low", "robustness_bonus": 0.04},
        )
    )
    decision = PrimaryOptimizer().decide(
        run_id="agentrun-test",
        turn_index=4,
        spec=spec,
        context=context,
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "compare_candidates"


@pytest.mark.unit
def test_primary_optimizer_uses_risk_aware_experiment_variant() -> None:
    """
    Verify elevated probe risk shifts next experiment request into risk-aware mode.
    """
    spec = _build_spec()
    context = ContextStore()
    context.record_tool_result(
        _tool_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-a", "score": 0.50},
        )
    )
    context.record_tool_result(
        _tool_result(
            turn_index=1,
            tool_name="run_probe_suite",
            output={"candidate_id": "cand-a", "risk_level": "high", "robustness_bonus": -0.02},
        )
    )
    decision = PrimaryOptimizer().decide(
        run_id="agentrun-test",
        turn_index=2,
        spec=spec,
        context=context,
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "run_experiment"
    assert decision.tool_arguments.get("variant_label") == "risk_aware"
    overrides = decision.tool_arguments.get("overrides")
    assert isinstance(overrides, dict)
    assert overrides["evaluation_episodes"] == 6


@pytest.mark.unit
def test_primary_optimizer_switches_to_cost_aware_mode_on_high_api_usage() -> None:
    """
    Verify high API usage shifts exploration payload to cost-aware settings.
    """
    spec = _build_spec()
    context = ContextStore()
    budget_state = _budget_state()
    budget_state.usage.api_cost_usd = 8.0
    decision = PrimaryOptimizer().decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=context,
        budget_state=budget_state,
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "run_experiment"
    assert decision.tool_arguments.get("variant_label") == "cost_aware"
    overrides = decision.tool_arguments.get("overrides")
    assert isinstance(overrides, dict)
    assert overrides["evaluation_episodes"] == 4


@pytest.mark.unit
def test_primary_optimizer_uses_planner_decision_when_available() -> None:
    """
    Verify optimizer accepts planner-authored decisions and wraps stop with export.
    """

    class _StubPlanner(_PlannerStubBase):
        """
        Return a deterministic stop decision for planner integration test.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint

            return AgentDecision(
                turn_index=turn_index,
                action=AgentDecisionAction.STOP,
                summary="planner stop",
                stop_tag=StopDecisionTag.MANUAL,
                stop_reason="planner requested stop",
            )

    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(update={"planner_provider": "openai"}),
        }
    )
    decision = PrimaryOptimizer(decision_planner=_StubPlanner()).decide(
        run_id="agentrun-test",
        turn_index=5,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "export_report"


@pytest.mark.unit
def test_primary_optimizer_normalizes_planner_run_experiment_arguments() -> None:
    """
    Verify planner run_experiment aliases are normalized into valid tool arguments.
    """

    class _StubPlanner(_PlannerStubBase):
        """
        Return a planner-style run_experiment decision with alias arguments.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=turn_index,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner experiment",
                tool_name="run_experiment",
                tool_arguments={
                    "environment": "gymnasium/Humanoid-v4",
                    "candidate_name": "baseline",
                    "training_timesteps": 100000,
                    "evaluation_episodes": 10,
                },
                tool_rationale="collect first score",
                decision_source="llm_openai",
            )

    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(update={"planner_provider": "openai"}),
        }
    )
    decision = PrimaryOptimizer(decision_planner=_StubPlanner()).decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "run_experiment"
    args = decision.tool_arguments
    assert args.get("environment_id") == "Humanoid-v4"
    assert args.get("environment_backend") == "gymnasium"
    assert args.get("candidate_id") == "baseline"
    overrides = args.get("overrides")
    assert isinstance(overrides, dict)
    assert overrides["ppo_total_timesteps"] == 100000
    assert overrides["evaluation_episodes"] == 10


@pytest.mark.unit
def test_primary_optimizer_normalizes_compound_environment_id_argument() -> None:
    """
    Verify planner environment_id aliases like backend/id are normalized.
    """

    class _StubPlanner(_PlannerStubBase):
        """
        Return run_experiment decision with compound environment_id input.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner experiment",
                tool_name="run_experiment",
                tool_arguments={"environment_id": "gymnasium/Humanoid-v4"},
                tool_rationale="collect evidence",
                decision_source="llm_openai",
            )

    spec = _build_spec().model_copy(
        update={"agent": _build_spec().agent.model_copy(update={"planner_provider": "openai"})}
    )
    decision = PrimaryOptimizer(decision_planner=_StubPlanner()).decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    args = decision.tool_arguments
    assert args["environment_backend"] == "gymnasium"
    assert args["environment_id"] == "Humanoid-v4"


@pytest.mark.unit
def test_primary_optimizer_ignores_planner_objective_file_override() -> None:
    """
    Verify run_experiment normalization pins objective/reward files to spec defaults.
    """

    class _StubPlanner(_PlannerStubBase):
        """
        Return planner decision attempting to override critical file paths.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner experiment with bad paths",
                tool_name="run_experiment",
                tool_arguments={
                    "objective_file": "objective.py",
                    "reward_file": "reward.py",
                },
                tool_rationale="collect evidence",
                decision_source="llm_openai",
            )

    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(update={"planner_provider": "openai"}),
        }
    )
    decision = PrimaryOptimizer(decision_planner=_StubPlanner()).decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    args = decision.tool_arguments
    assert args["objective_file"] == spec.objective.text_file
    assert args["reward_file"] == spec.objective.baseline_reward_file


@pytest.mark.unit
def test_primary_optimizer_stops_in_strict_mode_when_planner_unavailable() -> None:
    """
    Verify strict planner mode does not fall back to heuristic behavior.
    """

    class _UnavailablePlanner(_PlannerStubBase):
        """
        Simulate planner outage by returning no decision.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> None:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return None

    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(
                update={
                    "planner_provider": "openai",
                    "planner_fallback_enabled": False,
                }
            ),
            "tools": spec.tools.model_copy(
                update={"enabled": ("run_experiment", "run_probe_suite", "compare_candidates")}
            ),
        }
    )
    optimizer = PrimaryOptimizer(decision_planner=_UnavailablePlanner())
    decision = optimizer.decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    feedback = optimizer.drain_planner_feedback()
    assert decision.action is AgentDecisionAction.STOP
    assert decision.stop_reason == "planner unavailable and planner_fallback_enabled is false"
    assert any(row.failure_type == "planner_unavailable" for row in feedback)


@pytest.mark.unit
def test_primary_optimizer_applies_planner_usage_to_budget_state() -> None:
    """
    Verify planner token/cost/model-call usage is tracked in budget state.
    """

    class _UsagePlanner(_PlannerStubBase):
        """
        Return one planner decision and usage payload.
        """

        def __init__(self) -> None:
            self._usage = PlannerCallUsage(
                model_used="gpt-5.4-mini",
                api_input_tokens=123,
                api_output_tokens=45,
                api_cost_usd=0.67,
                call_count=1,
            )

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner experiment",
                tool_name="run_experiment",
                tool_arguments={},
                tool_rationale="collect evidence",
                decision_source="llm_openai",
            )

        def last_usage(self) -> PlannerCallUsage:
            return self._usage

    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(
                update={"planner_provider": "openai", "planner_fallback_enabled": True}
            )
        }
    )
    budget_state = _budget_state()
    _ = PrimaryOptimizer(decision_planner=_UsagePlanner()).decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=budget_state,
    )
    assert budget_state.usage.api_input_tokens == 123
    assert budget_state.usage.api_output_tokens == 45
    assert budget_state.usage.api_cost_usd == pytest.approx(0.67)
    assert budget_state.usage.calls_per_model["gpt-5.4-mini"] >= 1


@pytest.mark.unit
def test_primary_optimizer_stops_when_planner_api_budget_is_exhausted() -> None:
    """
    Verify OpenAI planner mode stops when planner API budgets are exhausted.
    """
    spec = _build_spec()
    spec = spec.model_copy(
        update={
            "agent": spec.agent.model_copy(
                update={"planner_provider": "openai", "planner_fallback_enabled": True}
            )
        }
    )
    budget_state = _budget_state()
    budget_state.usage.api_input_tokens = budget_state.max_api_input_tokens
    decision = PrimaryOptimizer().decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=budget_state,
    )
    assert decision.action is AgentDecisionAction.STOP
    assert decision.stop_reason == "planner API budget exhausted"


@pytest.mark.unit
def test_primary_optimizer_records_feedback_when_normalization_fails() -> None:
    """
    Verify optimizer records planner feedback when tool args cannot be normalized.
    """

    class _InvalidProbePlanner(_PlannerStubBase):
        """
        Return a probe request missing required arguments.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner probe",
                tool_name="run_probe_suite",
                tool_arguments={},
                tool_rationale="probe now",
                decision_source="llm_openai",
            )

    spec = _build_spec().model_copy(
        update={"agent": _build_spec().agent.model_copy(update={"planner_provider": "openai"})}
    )
    optimizer = PrimaryOptimizer(decision_planner=_InvalidProbePlanner())
    decision = optimizer.decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    feedback = optimizer.drain_planner_feedback()
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert any(row.failure_type == "invalid_tool_arguments" for row in feedback)


@pytest.mark.unit
def test_primary_optimizer_ignores_planner_probe_file_overrides() -> None:
    """
    Verify run_probe_suite normalization pins objective/reward files to spec defaults.
    """

    class _ProbePlanner(_PlannerStubBase):
        """
        Return probe decision with invalid objective/reward file overrides.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="planner probe",
                tool_name="run_probe_suite",
                tool_arguments={
                    "candidate_id": "cand-x",
                    "primary_score": 1.0,
                    "objective_file": "objective.txt",
                    "reward_file": "reward.py",
                },
                tool_rationale="probe robustness",
                decision_source="llm_openai",
            )

    spec = _build_spec().model_copy(
        update={"agent": _build_spec().agent.model_copy(update={"planner_provider": "openai"})}
    )
    decision = PrimaryOptimizer(decision_planner=_ProbePlanner()).decide(
        run_id="agentrun-test",
        turn_index=0,
        spec=spec,
        context=ContextStore(),
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "run_probe_suite"
    args = decision.tool_arguments
    assert args["objective_file"] == spec.objective.text_file
    assert args["reward_file"] == spec.objective.baseline_reward_file


@pytest.mark.unit
def test_primary_optimizer_replaces_planner_compare_candidates_payload() -> None:
    """
    Verify compare-candidates planner payload is replaced with normalized snapshots.
    """

    class _ComparePlanner(_PlannerStubBase):
        """
        Return compare decision with malformed candidate payload.
        """

        def plan(
            self,
            *,
            turn_index: int,
            spec: AgenticRunSpec,
            context: ContextStore,
            budget_state: BudgetState,
            stop_hint: object,
        ) -> AgentDecision:
            _ = turn_index
            _ = spec
            _ = context
            _ = budget_state
            _ = stop_hint
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="compare now",
                tool_name="compare_candidates",
                tool_arguments={"candidates": [{"bad": "shape"}]},
                tool_rationale="rank candidates",
                decision_source="llm_openai",
            )

    spec = _build_spec().model_copy(
        update={"agent": _build_spec().agent.model_copy(update={"planner_provider": "openai"})}
    )
    context = ContextStore()
    context.record_tool_result(
        _tool_result(
            turn_index=0,
            tool_name="run_experiment",
            output={"candidate_id": "cand-a", "score": 1.23},
        )
    )
    decision = PrimaryOptimizer(decision_planner=_ComparePlanner()).decide(
        run_id="agentrun-test",
        turn_index=1,
        spec=spec,
        context=context,
        budget_state=_budget_state(),
    )
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "compare_candidates"
    candidates = decision.tool_arguments["candidates"]
    assert isinstance(candidates, list)
    assert candidates and candidates[0]["candidate_id"] == "cand-a"
