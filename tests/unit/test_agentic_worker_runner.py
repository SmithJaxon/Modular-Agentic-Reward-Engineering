"""
Summary: Unit tests for worker packet construction and worker-runner execution.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from typing import Any

import pytest

from rewardlab.agentic.worker_context import WorkerTaskPacket, build_worker_task_packet
from rewardlab.agentic.worker_runner import WorkerRunner
from rewardlab.schemas.agentic_run import (
    AgentDecision,
    AgentDecisionAction,
    AgenticRunSpec,
    AgentProfileSpec,
    BudgetSpec,
    DecisionPolicySpec,
    EnvironmentSpec,
    HardBudgetSpec,
    ObjectiveSpec,
    ReportingSpec,
    SoftBudgetGuidanceSpec,
    ToolPolicySpec,
    TrainingDefaultsSpec,
)
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.schemas.tool_contracts import ToolResult, ToolResultStatus


class _FakeBroker:
    """
    Capture execute calls and return a fixed successful tool result.
    """

    def __init__(self) -> None:
        """
        Initialize capture state.
        """
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        *,
        turn_index: int,
        tool_name: str,
        arguments: dict[str, object],
        rationale: str,
    ) -> ToolResult:
        """
        Record call payload and return a deterministic completed result.
        """
        self.calls.append(
            {
                "turn_index": turn_index,
                "tool_name": tool_name,
                "arguments": arguments,
                "rationale": rationale,
            }
        )
        return ToolResult(
            request_id="toolreq-worker",
            turn_index=turn_index,
            tool_name=tool_name,
            status=ToolResultStatus.COMPLETED,
            output={"score": 1.0},
        )


def _spec() -> AgenticRunSpec:
    """
    Build a minimal valid run spec for worker packet testing.
    """
    return AgenticRunSpec(
        version=1,
        run_name="worker-test",
        environment=EnvironmentSpec(
            backend=EnvironmentBackend.GYMNASIUM,
            id="CartPole-v1",
            seed=7,
        ),
        objective=ObjectiveSpec(
            text_file="tools/fixtures/objectives/cartpole.txt",
            baseline_reward_file="tools/fixtures/rewards/cartpole_baseline.py",
        ),
        agent=AgentProfileSpec(
            primary_model="gpt-5.4-mini",
            fallback_model="gpt-4o-mini",
        ),
        tools=ToolPolicySpec(enabled=("run_experiment",)),
        decision=DecisionPolicySpec(max_turns=5),
        budgets=BudgetSpec(
            hard=HardBudgetSpec(
                max_wall_clock_minutes=10,
                max_training_timesteps=1000,
                max_evaluation_episodes=100,
                max_api_input_tokens=1000,
                max_api_output_tokens=1000,
                max_api_usd=1.0,
                max_calls_per_model={},
            ),
            soft=SoftBudgetGuidanceSpec(),
        ),
        training_defaults=TrainingDefaultsSpec(),
        reporting=ReportingSpec(),
    )


@pytest.mark.unit
def test_build_worker_task_packet_produces_minimal_packet() -> None:
    """
    Verify worker packets contain required execution fields only.
    """
    decision = AgentDecision(
        turn_index=2,
        action=AgentDecisionAction.REQUEST_TOOL,
        summary="run experiment",
        tool_name="run_experiment",
        tool_arguments={"environment_id": "CartPole-v1"},
        tool_rationale="test rationale",
    )
    packet = build_worker_task_packet(
        run_id="agentrun-test",
        decision=decision,
        spec=_spec(),
        budget_remaining={"remaining_training_timesteps": 1000},
    )
    assert isinstance(packet, WorkerTaskPacket)
    assert packet.run_id == "agentrun-test"
    assert packet.turn_index == 2
    assert packet.tool_name == "run_experiment"
    assert packet.tool_arguments["environment_id"] == "CartPole-v1"
    assert packet.objective_hint == "worker-test"


@pytest.mark.unit
def test_worker_runner_executes_packet_through_broker() -> None:
    """
    Verify worker runner forwards packet content and returns execution metadata.
    """
    broker = _FakeBroker()
    runner = WorkerRunner(tool_broker=broker)  # type: ignore[arg-type]
    packet = WorkerTaskPacket(
        run_id="agentrun-test",
        turn_index=1,
        tool_name="budget_snapshot",
        tool_arguments={"x": 1},
        tool_rationale="check budget",
        objective_hint="worker-test",
        budget_remaining={"remaining_training_timesteps": 10},
        requested_at="2026-04-10T00:00:00+00:00",
    )
    record = runner.execute(packet)
    assert broker.calls
    assert broker.calls[0]["tool_name"] == "budget_snapshot"
    assert record.result.status is ToolResultStatus.COMPLETED
    assert record.packet.turn_index == 1
    assert record.started_at <= record.finished_at
