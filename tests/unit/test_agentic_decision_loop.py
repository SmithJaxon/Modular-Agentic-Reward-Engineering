"""
Summary: Unit tests for planner feedback event/report handling in decision loop.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.decision_loop import DecisionLoop
from rewardlab.agentic.llm_planner import PlannerAttemptFeedback
from rewardlab.agentic.worker_runner import WorkerExecutionRecord
from rewardlab.persistence.agentic_repository import AgenticRepository
from rewardlab.schemas.agentic_run import (
    AgentDecision,
    AgentDecisionAction,
    AgenticRunSpec,
    StopDecisionTag,
)
from rewardlab.schemas.tool_contracts import ToolResult, ToolResultStatus


def _spec() -> AgenticRunSpec:
    """
    Build a minimal valid spec for decision-loop tests.
    """
    return AgenticRunSpec.model_validate(
        {
            "version": 1,
            "run_name": "decision-loop-test",
            "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 7},
            "objective": {
                "text_file": "tools/fixtures/objectives/cartpole.txt",
                "baseline_reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
            },
            "agent": {
                "primary_model": "gpt-5.4-mini",
                "reasoning_effort": "high",
                "fallback_model": "gpt-4o-mini",
                "planner_provider": "openai",
            },
            "tools": {"enabled": ["run_experiment"], "max_parallel_workers": 1},
            "decision": {"max_turns": 3},
            "budgets": {
                "hard": {
                    "max_wall_clock_minutes": 10,
                    "max_training_timesteps": 10000,
                    "max_evaluation_episodes": 100,
                    "max_api_input_tokens": 5000,
                    "max_api_output_tokens": 5000,
                    "max_api_usd": 5.0,
                    "max_calls_per_model": {"gpt-5.4-mini": 10},
                }
            },
        }
    )


class _StubPrimaryOptimizer:
    """
    Return deterministic decisions and one planner-feedback row.
    """

    def __init__(self) -> None:
        """
        Initialize turn counter and feedback queue.
        """
        self._turn = 0
        self._feedback: tuple[PlannerAttemptFeedback, ...] = (
            PlannerAttemptFeedback(
                attempt_index=1,
                max_attempts=2,
                failure_type="parse_error",
                reason="response was not a valid JSON object",
                output_excerpt="not json",
            ),
        )

    def decide(
        self,
        *,
        run_id: str,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: object,
    ) -> AgentDecision:
        """
        Return one tool-call decision followed by one stop decision.
        """
        _ = run_id
        _ = turn_index
        _ = spec
        _ = context
        _ = budget_state
        if self._turn == 0:
            self._turn += 1
            return AgentDecision(
                turn_index=0,
                action=AgentDecisionAction.REQUEST_TOOL,
                summary="run one experiment",
                tool_name="run_experiment",
                tool_arguments={
                    "session_id": "agentrun-test",
                    "candidate_id": "cand-a",
                    "environment_id": "CartPole-v1",
                    "environment_backend": "gymnasium",
                    "objective_file": "tools/fixtures/objectives/cartpole.txt",
                    "reward_file": "tools/fixtures/rewards/cartpole_baseline.py",
                    "seed": 7,
                    "iteration_index": 0,
                    "variant_label": "default",
                    "include_reflection": False,
                    "overrides": {
                        "execution_mode": "deterministic",
                        "llm_provider": "none",
                        "llm_model": "gpt-5.4-mini",
                        "ppo_total_timesteps": 256,
                        "ppo_num_envs": 1,
                        "ppo_n_steps": 64,
                        "ppo_batch_size": 64,
                        "ppo_learning_rate": 0.0003,
                        "evaluation_episodes": 2,
                        "reflection_episodes": 0,
                        "reflection_interval_steps": 128,
                        "train_seed": 7,
                    },
                },
                tool_rationale="collect evidence",
                decision_source="llm_openai",
            )
        return AgentDecision(
            turn_index=1,
            action=AgentDecisionAction.STOP,
            summary="finished",
            stop_tag=StopDecisionTag.MANUAL,
            stop_reason="test stop",
        )

    def drain_planner_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return feedback rows once, then clear.
        """
        rows = self._feedback
        self._feedback = ()
        return rows


class _StubWorkerRunner:
    """
    Return deterministic completed tool results for decision-loop testing.
    """

    def execute(self, packet: object) -> WorkerExecutionRecord:
        """
        Build one successful execution record from the incoming packet.
        """
        _ = packet
        result = ToolResult(
            request_id="toolreq-test",
            turn_index=0,
            tool_name="run_experiment",
            status=ToolResultStatus.COMPLETED,
            output={"candidate_id": "cand-a", "score": 0.75},
        )
        return WorkerExecutionRecord(
            packet=packet,  # type: ignore[arg-type]
            started_at="2026-04-11T00:00:00+00:00",
            finished_at="2026-04-11T00:00:01+00:00",
            result=result,
        )


@pytest.mark.unit
def test_decision_loop_persists_planner_feedback_events_and_report_summary(
) -> None:
    """
    Verify planner feedback rows are emitted as events and summarized in reports.
    """
    spec = _spec()
    temp_root = Path(f"pytesttmp-agentic-loop-{uuid4().hex[:8]}")
    temp_root.mkdir(parents=True, exist_ok=True)
    repository = AgenticRepository(temp_root)
    run_id = "agentrun-feedback-test"
    try:
        repository.create_run(
            run_id=run_id,
            spec_payload=spec.model_dump(mode="json"),
            spec_path="configs/agentic/humanoid_main_openai.yaml",
        )
        loop = DecisionLoop(
            primary_optimizer=_StubPrimaryOptimizer(),  # type: ignore[arg-type]
            worker_runner=_StubWorkerRunner(),  # type: ignore[arg-type]
            repository=repository,
        )
        result = loop.execute(
            run_id=run_id,
            spec=spec,
            context=ContextStore(),
            budget_engine=BudgetEngine.from_spec(spec),
        )
        events = repository.list_events(run_id)
        planner_events = [
            row
            for row in events
            if isinstance(row, dict) and row.get("event_type") == "planner.validation_failed"
        ]
        assert result.status.value == "completed"
        assert len(planner_events) == 1
        summary = result.report_payload["planner_feedback_summary"]
        assert summary["failure_count"] == 1
        assert summary["by_failure_type"]["parse_error"] == 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
