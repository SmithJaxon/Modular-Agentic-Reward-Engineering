"""
Summary: Unit tests for the LLM-driven primary decision planner.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.llm_planner import LLMDecisionPlanner
from rewardlab.llm.openai_client import OpenAITextResponse
from rewardlab.schemas.agentic_run import AgentDecisionAction, AgenticRunSpec
from rewardlab.schemas.budget_state import BudgetState


@dataclass(slots=True)
class _FakeClient:
    """
    Return a fixed planner response for deterministic unit testing.
    """

    response_text: str

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Return the configured response and ignore generation arguments.
        """
        _ = prompt
        _ = max_output_tokens
        _ = reasoning_effort
        return self.response_text


@dataclass(slots=True)
class _FakeUsageClient:
    """
    Return a fixed planner response plus usage metrics.
    """

    response_text: str

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Return fixed planner text for protocol compatibility.
        """
        _ = prompt
        _ = max_output_tokens
        _ = reasoning_effort
        return self.response_text

    def generate_text_with_usage(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> OpenAITextResponse:
        """
        Return fixed planner text and usage values.
        """
        _ = prompt
        _ = max_output_tokens
        _ = reasoning_effort
        return OpenAITextResponse(
            text=self.response_text,
            model_used="gpt-5.4-mini",
            api_input_tokens=111,
            api_output_tokens=22,
            api_cost_usd=0.33,
        )


@dataclass(slots=True)
class _FakeSequenceClient:
    """
    Return sequential planner responses across repeated planner attempts.
    """

    responses: list[str]
    call_index: int = 0

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Return the next configured response, repeating the last when exhausted.
        """
        _ = prompt
        _ = max_output_tokens
        _ = reasoning_effort
        if not self.responses:
            return ""
        if self.call_index >= len(self.responses):
            return self.responses[-1]
        value = self.responses[self.call_index]
        self.call_index += 1
        return value


def _spec(*, planner_provider: str) -> AgenticRunSpec:
    """
    Build a minimal spec for LLM planner behavior tests.
    """
    return AgenticRunSpec.model_validate(
        {
            "version": 1,
            "run_name": "planner-test",
            "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 7},
            "objective": {
                "text_file": "tools/fixtures/objectives/cartpole.txt",
                "baseline_reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
            },
            "agent": {
                "primary_model": "gpt-5.4-mini",
                "reasoning_effort": "high",
                "fallback_model": "gpt-4o-mini",
                "planner_provider": planner_provider,
                "planner_context_window": 4,
                "planner_max_output_tokens": 600,
                "planner_max_retries": 1,
            },
            "tools": {
                "enabled": ["run_experiment", "export_report"],
                "max_parallel_workers": 1,
                "per_call_timeout_seconds": 120,
            },
            "decision": {"max_turns": 10},
            "budgets": {
                "hard": {
                    "max_wall_clock_minutes": 15,
                    "max_training_timesteps": 10000,
                    "max_evaluation_episodes": 50,
                    "max_api_input_tokens": 50000,
                    "max_api_output_tokens": 50000,
                    "max_api_usd": 5.0,
                    "max_calls_per_model": {"gpt-5.4-mini": 20},
                }
            },
        }
    )


def _budget_state() -> BudgetState:
    """
    Build a permissive budget state for planner tests.
    """
    return BudgetState(
        max_wall_clock_minutes=15,
        max_training_timesteps=10000,
        max_evaluation_episodes=50,
        max_api_input_tokens=50000,
        max_api_output_tokens=50000,
        max_api_usd=5.0,
        max_calls_per_model={"gpt-5.4-mini": 20},
    )


@pytest.mark.unit
def test_llm_planner_returns_request_tool_decision_from_json() -> None:
    """
    Verify valid JSON planner output is parsed into an agent decision.
    """
    planner = LLMDecisionPlanner(
        client_factory=lambda _: _FakeClient(
            """
            {
              "action": "request_tool",
              "summary": "Run one exploration experiment.",
              "tool_name": "run_experiment",
              "tool_arguments": {"variant_label": "planner_test"},
              "tool_rationale": "Need fresh evidence from one candidate."
            }
            """
        )
    )
    decision = planner.plan(
        turn_index=3,
        spec=_spec(planner_provider="openai"),
        context=ContextStore(),
        budget_state=_budget_state(),
        stop_hint=None,
    )
    assert decision is not None
    assert decision.action is AgentDecisionAction.REQUEST_TOOL
    assert decision.tool_name == "run_experiment"
    assert decision.turn_index == 3


@pytest.mark.unit
def test_llm_planner_rejects_tool_not_in_allowlist() -> None:
    """
    Verify planner outputs are dropped when they request a non-enabled tool.
    """
    planner = LLMDecisionPlanner(
        client_factory=lambda _: _FakeClient(
            """
            {
              "action": "request_tool",
              "summary": "Call unknown tool.",
              "tool_name": "unknown_tool",
              "tool_arguments": {},
              "tool_rationale": "Not allowed."
            }
            """
        )
    )
    decision = planner.plan(
        turn_index=0,
        spec=_spec(planner_provider="openai"),
        context=ContextStore(),
        budget_state=_budget_state(),
        stop_hint=None,
    )
    feedback = planner.last_feedback()
    assert decision is None
    assert len(feedback) == 2
    assert all(row.failure_type == "tool_not_enabled" for row in feedback)


@pytest.mark.unit
def test_llm_planner_tracks_last_usage_when_client_exposes_usage() -> None:
    """
    Verify planner records usage for budget accounting when available.
    """
    planner = LLMDecisionPlanner(
        client_factory=lambda _: _FakeUsageClient(
            """
            {
              "action": "request_tool",
              "summary": "Run one experiment.",
              "tool_name": "run_experiment",
              "tool_arguments": {},
              "tool_rationale": "Gather evidence."
            }
            """
        )
    )
    decision = planner.plan(
        turn_index=2,
        spec=_spec(planner_provider="openai"),
        context=ContextStore(),
        budget_state=_budget_state(),
        stop_hint=None,
    )
    usage = planner.last_usage()
    assert decision is not None
    assert usage is not None
    assert usage.api_input_tokens == 111
    assert usage.api_output_tokens == 22
    assert usage.api_cost_usd == pytest.approx(0.33)


@pytest.mark.unit
def test_llm_planner_returns_none_when_provider_is_not_openai() -> None:
    """
    Verify planner is bypassed for non-openai planner providers.
    """
    planner = LLMDecisionPlanner(
        client_factory=lambda _: _FakeClient(
            """
            {
              "action": "stop",
              "summary": "stop",
              "stop_tag": "manual",
              "stop_reason": "provider bypass test"
            }
            """
        )
    )
    decision = planner.plan(
        turn_index=1,
        spec=_spec(planner_provider="heuristic"),
        context=ContextStore(),
        budget_state=_budget_state(),
        stop_hint=None,
    )
    assert decision is None


@pytest.mark.unit
def test_llm_planner_retries_invalid_output_and_returns_second_valid_attempt() -> None:
    """
    Verify planner retries after malformed output and succeeds on a later attempt.
    """
    sequence_client = _FakeSequenceClient(
        responses=[
            "this is not json",
            """
            {
              "action": "request_tool",
              "summary": "Retry with valid JSON.",
              "tool_name": "run_experiment",
              "tool_arguments": {},
              "tool_rationale": "Collect evidence."
            }
            """,
        ]
    )
    planner = LLMDecisionPlanner(client_factory=lambda _: sequence_client)
    decision = planner.plan(
        turn_index=4,
        spec=_spec(planner_provider="openai"),
        context=ContextStore(),
        budget_state=_budget_state(),
        stop_hint=None,
    )
    usage = planner.last_usage()
    feedback = planner.last_feedback()
    assert decision is not None
    assert decision.tool_name == "run_experiment"
    assert usage is not None
    assert usage.call_count == 2
    assert len(feedback) == 1
    assert feedback[0].failure_type == "parse_error"
