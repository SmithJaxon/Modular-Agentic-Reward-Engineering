"""
Summary: Unit tests for autonomous controller decision behavior.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rewardlab.agentic.controller import ControllerAgent, ControllerContext
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.schemas.agent_experiment import (
    AgentBudgetLedger,
    AgentExperimentRecord,
    AgentExperimentSpec,
    AgentExperimentStatus,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import FeedbackGate


class NoCredentialClient:
    """OpenAI client double that disables network-backed controller behavior."""

    has_credentials = False

    def chat_completion(self, request):  # pragma: no cover - should never be called
        """Fail fast if a test accidentally invokes network-backed completion."""

        del request
        raise AssertionError("chat_completion should not run without credentials")


def _record_from_fixture() -> AgentExperimentRecord:
    """Build a running experiment record from the balanced fixture spec."""

    spec = load_experiment_spec(Path("tools/fixtures/experiments/agent_humanoid_balanced.yaml"))
    return AgentExperimentRecord(
        experiment_id="experiment-test",
        status=AgentExperimentStatus.RUNNING,
        spec=spec,
        created_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        budget_ledger=AgentBudgetLedger(),
    )


def _completed_run(
    *,
    record: AgentExperimentRecord,
    run_id: str,
    candidate_id: str,
    reward: float,
) -> ExperimentRun:
    """Return a completed run fixture for one candidate."""

    return ExperimentRun(
        run_id=run_id,
        candidate_id=candidate_id,
        backend=record.spec.environment.backend,
        environment_id=record.spec.environment.id,
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        status=RunStatus.COMPLETED,
        metrics={"episode_reward": reward},
        artifact_refs=["a.json"],
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 4, 10, 12, 1, tzinfo=UTC),
    )


def test_controller_heuristic_runs_unevaluated_latest_candidate() -> None:
    """Fallback controller should run latest candidate when it has no run evidence."""

    record = _record_from_fixture()
    candidate = RewardCandidate(
        candidate_id="experiment-test-candidate-000",
        session_id="experiment-test",
        iteration_index=0,
        reward_definition="def reward(observation):\n    return 1.0\n",
        change_summary="baseline",
        aggregate_score=None,
    )
    action, tokens = ControllerAgent(openai_client=NoCredentialClient()).choose_action(
        ControllerContext(
            record=record,
            candidates=[candidate],
            runs=[],
            recent_decisions=[],
            failed_actions=0,
            no_improve_streak=0,
        )
    )

    assert tokens == 0
    assert action.action_type.value == "run_experiment"


def test_controller_heuristic_proposes_after_latest_is_evaluated() -> None:
    """Fallback controller should propose a new reward after latest run is scored."""

    record = _record_from_fixture()
    candidate = RewardCandidate(
        candidate_id="experiment-test-candidate-000",
        session_id="experiment-test",
        iteration_index=0,
        reward_definition="def reward(observation):\n    return 1.0\n",
        change_summary="baseline",
        aggregate_score=0.4,
    )
    run = _completed_run(
        record=record,
        run_id="experiment-test-run-001",
        candidate_id=candidate.candidate_id,
        reward=0.4,
    )

    action, _ = ControllerAgent(openai_client=NoCredentialClient()).choose_action(
        ControllerContext(
            record=record,
            candidates=[candidate],
            runs=[run],
            recent_decisions=[],
            failed_actions=0,
            no_improve_streak=0,
        )
    )

    assert action.action_type.value == "propose_reward"


def test_controller_heuristic_compares_candidates_on_stall() -> None:
    """Fallback controller should compare candidates when no-improve stalls."""

    record = _record_from_fixture()
    candidate_0 = RewardCandidate(
        candidate_id="experiment-test-candidate-000",
        session_id="experiment-test",
        iteration_index=0,
        reward_definition="def reward(observation):\n    return 1.0\n",
        change_summary="baseline",
        aggregate_score=1.0,
    )
    candidate_1 = RewardCandidate(
        candidate_id="experiment-test-candidate-001",
        session_id="experiment-test",
        parent_candidate_id=candidate_0.candidate_id,
        iteration_index=1,
        reward_definition="def reward(observation):\n    return 2.0\n",
        change_summary="variant",
        aggregate_score=0.95,
    )
    action, _ = ControllerAgent(openai_client=NoCredentialClient()).choose_action(
        ControllerContext(
            record=record,
            candidates=[candidate_0, candidate_1],
            runs=[
                _completed_run(
                    record=record,
                    run_id="experiment-test-run-001",
                    candidate_id=candidate_0.candidate_id,
                    reward=1.0,
                ),
                _completed_run(
                    record=record,
                    run_id="experiment-test-run-002",
                    candidate_id=candidate_1.candidate_id,
                    reward=0.95,
                ),
            ],
            recent_decisions=[],
            failed_actions=0,
            no_improve_streak=1,
        )
    )

    assert action.action_type.value == "compare_candidates"


def test_controller_heuristic_requests_human_feedback_when_enabled() -> None:
    """Fallback controller should request human feedback when policy allows it."""

    original = _record_from_fixture()
    payload = original.spec.model_dump(mode="python")
    payload["tool_policy"]["allowed_tools"].append("request_human_feedback")
    payload["governance"]["human_feedback"] = {
        "allow": True,
        "feedback_gate": FeedbackGate.ONE_REQUIRED.value,
        "max_requests": 2,
    }
    spec = AgentExperimentSpec.model_validate(payload)
    record = original.model_copy(update={"spec": spec})

    candidate = RewardCandidate(
        candidate_id="experiment-test-candidate-000",
        session_id="experiment-test",
        iteration_index=0,
        reward_definition="def reward(observation):\n    return 1.0\n",
        change_summary="baseline",
        aggregate_score=0.4,
    )
    action, _ = ControllerAgent(openai_client=NoCredentialClient()).choose_action(
        ControllerContext(
            record=record,
            candidates=[candidate],
            runs=[
                _completed_run(
                    record=record,
                    run_id="experiment-test-run-001",
                    candidate_id=candidate.candidate_id,
                    reward=0.4,
                )
            ],
            recent_decisions=[],
            failed_actions=0,
            no_improve_streak=1,
        )
    )

    assert action.action_type.value == "request_human_feedback"
