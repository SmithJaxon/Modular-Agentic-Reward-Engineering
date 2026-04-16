"""
Summary: Unit tests for autonomous analysis and feedback-related worker tools.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.agentic.tools.compare_candidates import CompareCandidatesTool
from rewardlab.agentic.tools.estimate_cost_and_risk import EstimateCostAndRiskTool
from rewardlab.agentic.tools.summarize_run_artifacts import SummarizeRunArtifactsTool
from rewardlab.agentic.tools.validate_reward_program import ValidateRewardProgramTool
from rewardlab.llm.openai_client import ChatCompletionResponse
from rewardlab.schemas.agent_experiment import (
    AgentBudgetLedger,
    AgentExperimentRecord,
    AgentExperimentStatus,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate


class FakeAnalyzerClient:
    """OpenAI client double with deterministic analyzer JSON responses."""

    def __init__(self, response_content: str, total_tokens: int = 21) -> None:
        """Store synthetic response payload and token usage."""

        self.has_credentials = True
        self._response_content = response_content
        self._total_tokens = total_tokens

    def chat_completion(self, request):  # pragma: no cover - request shape is not asserted
        """Return a canned analyzer response."""

        del request
        return ChatCompletionResponse(
            content=self._response_content,
            raw_response=None,
            total_tokens=self._total_tokens,
        )


def _record_from_fixture() -> AgentExperimentRecord:
    """Build a running experiment record from the low-cost CartPole fixture."""

    spec = load_experiment_spec(Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml"))
    return AgentExperimentRecord(
        experiment_id="experiment-tool-tests",
        status=AgentExperimentStatus.RUNNING,
        spec=spec,
        created_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        budget_ledger=AgentBudgetLedger(consumed_total_tokens=100),
    )


def test_compare_candidates_uses_analyzer_response() -> None:
    """Compare tool should adopt analyzer recommendation and token accounting."""

    record = _record_from_fixture()
    candidates = [
        RewardCandidate(
            candidate_id="experiment-tool-tests-candidate-000",
            session_id=record.experiment_id,
            iteration_index=0,
            reward_definition="def reward(observation): return 1.0",
            change_summary="baseline",
            aggregate_score=0.8,
        ),
        RewardCandidate(
            candidate_id="experiment-tool-tests-candidate-001",
            session_id=record.experiment_id,
            parent_candidate_id="experiment-tool-tests-candidate-000",
            iteration_index=1,
            reward_definition="def reward(observation): return 2.0",
            change_summary="variant",
            aggregate_score=0.7,
        ),
    ]
    tool = CompareCandidatesTool(
        openai_client=FakeAnalyzerClient(
            (
                '{"recommended_candidate_id":"experiment-tool-tests-candidate-001",'
                '"summary":"lower variance"}'
            ),
            total_tokens=42,
        )
    )

    result = tool.execute(
        record=record,
        candidates=candidates,
        runs=[],
        action_input={},
    )

    assert result.status == "ok"
    assert result.payload["recommended_candidate_id"] == "experiment-tool-tests-candidate-001"
    assert result.consumed_tokens == 42


def test_estimate_cost_and_risk_uses_analyzer_response() -> None:
    """Risk tool should expose analyzer-adjusted risk level and token usage."""

    record = _record_from_fixture()
    tool = EstimateCostAndRiskTool(
        openai_client=FakeAnalyzerClient(
            '{"risk_level":"high","recommend_stop":true,"summary":"budget nearly exhausted"}',
            total_tokens=33,
        )
    )

    result = tool.execute(
        record=record,
        candidates=[],
        runs=[],
        action_input={},
    )

    assert result.status == "ok"
    assert result.payload["risk_level"] == "high"
    assert result.payload["recommend_stop"] is True
    assert result.consumed_tokens == 33


def test_summarize_run_artifacts_returns_latest_run_summary() -> None:
    """Summarizer should return compact score/metric evidence for the latest run."""

    record = _record_from_fixture()
    run = ExperimentRun(
        run_id="experiment-tool-tests-run-001",
        candidate_id="experiment-tool-tests-candidate-000",
        backend=record.spec.environment.backend,
        environment_id=record.spec.environment.id,
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        status=RunStatus.COMPLETED,
        ended_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
        metrics={"episode_reward": 1.25, "step_count": 123},
        artifact_refs=["manifest.json", "metrics.json"],
    )
    result = SummarizeRunArtifactsTool().execute(
        record=record,
        candidates=[],
        runs=[run],
        action_input={},
    )

    assert result.status == "ok"
    assert result.payload["run_id"] == run.run_id
    assert result.payload["score"] == 1.25


def test_validate_reward_program_accepts_valid_candidate_source() -> None:
    """Validation tool should return executable signature details for valid reward code."""

    record = _record_from_fixture()
    candidate = RewardCandidate(
        candidate_id="experiment-tool-tests-candidate-000",
        session_id=record.experiment_id,
        iteration_index=0,
        reward_definition="def compute_reward(observation):\n    return 1.0\n",
        change_summary="baseline",
    )
    result = ValidateRewardProgramTool().execute(
        record=record,
        candidates=[candidate],
        runs=[],
        action_input={},
    )

    assert result.status == "ok"
    assert result.payload["validation_status"] == "valid"
    assert result.payload["entrypoint_name"] == "compute_reward"
