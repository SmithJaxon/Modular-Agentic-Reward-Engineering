"""
Summary: Unit tests for agentic tool broker retry and timeout semantics.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.agentic.tool_broker import ToolBroker
from rewardlab.schemas.agent_experiment import (
    ActionType,
    AgentExperimentRecord,
    AgentExperimentStatus,
)
from rewardlab.schemas.reward_candidate import RewardCandidate


@dataclass
class StaticTool:
    """Tool double returning a static result payload."""

    result: ToolResult

    def execute(self, **kwargs: object) -> ToolResult:
        """Return a fixed result for each execution."""

        del kwargs
        return self.result


@dataclass
class FailThenSucceedTool:
    """Tool double that fails once and succeeds on the second invocation."""

    attempts: int = 0

    def execute(self, **kwargs: object) -> ToolResult:
        """Return error on first call, success on subsequent calls."""

        del kwargs
        self.attempts += 1
        if self.attempts == 1:
            return ToolResult(status="error", summary="transient failure")
        return ToolResult(status="ok", summary="eventual success", payload={"done": True})


@dataclass
class SlowTool:
    """Tool double that sleeps longer than the configured timeout."""

    sleep_seconds: int

    def execute(self, **kwargs: object) -> ToolResult:
        """Sleep for configured duration, then return success."""

        del kwargs
        time.sleep(self.sleep_seconds)
        return ToolResult(status="ok", summary="finished late")


def _record_from_fixture() -> AgentExperimentRecord:
    """Create a running experiment record from the low-cost CartPole fixture."""

    spec = load_experiment_spec(Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml"))
    return AgentExperimentRecord(
        experiment_id="broker-tests",
        status=AgentExperimentStatus.RUNNING,
        spec=spec,
        created_at=datetime(2026, 4, 16, 12, 0, tzinfo=UTC),
        started_at=datetime(2026, 4, 16, 12, 0, tzinfo=UTC),
    )


def _candidate() -> RewardCandidate:
    """Return one baseline candidate fixture used by broker tests."""

    return RewardCandidate(
        candidate_id="broker-tests-candidate-000",
        session_id="broker-tests",
        iteration_index=0,
        reward_definition="def compute_reward(observation):\n    return 1.0\n",
        change_summary="baseline",
    )


def test_tool_broker_retries_and_returns_success_after_transient_failure() -> None:
    """Broker should retry failed local tools up to configured retry budget."""

    retry_tool = FailThenSucceedTool()
    record = _record_from_fixture().model_copy(
        update={
            "spec": _record_from_fixture().spec.model_copy(
                update={
                    "tool_policy": _record_from_fixture().spec.tool_policy.model_copy(
                        update={
                            "max_retries_per_tool": 1,
                            "default_timeout_seconds": 10,
                        }
                    )
                }
            )
        }
    )
    broker = ToolBroker(
        run_experiment_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        propose_reward_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        summarize_run_artifacts_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        validate_reward_program_tool=retry_tool,  # type: ignore[arg-type]
        estimate_cost_and_risk_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        compare_candidates_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        request_human_feedback_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
    )

    result = broker.execute_action(
        record=record,
        action=ControllerAction(
            action_type=ActionType.VALIDATE_REWARD_PROGRAM,
            rationale="validate before running",
        ),
        candidates=[_candidate()],
        runs=[],
    )

    assert result.status == "ok"
    assert result.payload["retry_attempts_used"] == 1


def test_tool_broker_enforces_timeout_for_local_tools() -> None:
    """Broker should fail when local execution exceeds tool timeout policy."""

    record = _record_from_fixture().model_copy(
        update={
            "spec": _record_from_fixture().spec.model_copy(
                update={
                    "tool_policy": _record_from_fixture().spec.tool_policy.model_copy(
                        update={
                            "default_timeout_seconds": 1,
                            "max_retries_per_tool": 0,
                        }
                    )
                }
            )
        }
    )
    broker = ToolBroker(
        run_experiment_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        propose_reward_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        summarize_run_artifacts_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        validate_reward_program_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        estimate_cost_and_risk_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
        compare_candidates_tool=SlowTool(sleep_seconds=2),  # type: ignore[arg-type]
        request_human_feedback_tool=StaticTool(ToolResult(status="ok", summary="noop")),  # type: ignore[arg-type]
    )

    result = broker.execute_action(
        record=record,
        action=ControllerAction(
            action_type=ActionType.COMPARE_CANDIDATES,
            rationale="compare candidates",
        ),
        candidates=[_candidate()],
        runs=[],
    )

    assert result.status == "error"
    assert "timed out" in result.summary
