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
from rewardlab.agentic.tools.run_experiment import RunExperimentTool
from rewardlab.agentic.tools.run_robustness_probes import RunRobustnessProbesTool
from rewardlab.agentic.tools.summarize_run_artifacts import SummarizeRunArtifactsTool
from rewardlab.agentic.tools.validate_reward_program import ValidateRewardProgramTool
from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import ExperimentExecutionService
from rewardlab.llm.openai_client import ChatCompletionResponse
from rewardlab.schemas.agent_experiment import (
    AgentBudgetLedger,
    AgentExperimentRecord,
    AgentExperimentStatus,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment


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


class FakeRobustnessRunner:
    """Runner double returning deterministic robustness probe outputs."""

    def run_candidate_probes(self, **kwargs):  # pragma: no cover - shape asserted by test
        """Return one completed robustness probe and a high-risk assessment."""

        candidate = kwargs["candidate"]
        primary_run = kwargs["primary_run"]
        probe_run = ExperimentRun(
            run_id=f"{candidate.candidate_id}-robustness-001",
            candidate_id=candidate.candidate_id,
            backend=primary_run.backend,
            environment_id=primary_run.environment_id,
            run_type=RunType.ROBUSTNESS,
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            variant_label="seed-17",
            status=RunStatus.COMPLETED,
            metrics={"episode_reward": 0.4},
            artifact_refs=["manifest.json", "metrics.json"],
            started_at=datetime(2026, 4, 10, 12, 6, tzinfo=UTC),
            ended_at=datetime(2026, 4, 10, 12, 7, tzinfo=UTC),
        )
        assessment = RobustnessAssessment(
            assessment_id=f"{candidate.candidate_id}-robustness",
            candidate_id=candidate.candidate_id,
            backend=primary_run.backend,
            primary_run_id=primary_run.run_id,
            probe_run_ids=[probe_run.run_id],
            variant_count=1,
            degradation_ratio=0.55,
            risk_level=RiskLevel.HIGH,
            risk_notes="Worst probe underperformed significantly.",
        )
        return [probe_run], assessment


class FakeExecutionResult:
    """Minimal execution-result wrapper returned by the fake execution service."""

    def __init__(self, run: ExperimentRun) -> None:
        """Store one completed or failed experiment run."""

        self.run = run


class FakeExecutionService:
    """Execution service double returning deterministic per-candidate run results."""

    def __init__(self) -> None:
        """Expose an artifact writer root compatible with RunExperimentTool."""

        self.artifact_writer = RunArtifactWriter(Path(".rewardlab") / "runs")

    def execute_candidate(self, **kwargs):  # pragma: no cover - shape asserted by test
        """Return a completed run for candidate ids ending with 001/002, else failed."""

        candidate = kwargs["candidate"]
        request = kwargs["request"]
        run = ExperimentRun(
            run_id=request.run_id,
            candidate_id=candidate.candidate_id,
            backend=request.backend,
            environment_id=request.environment_id,
            run_type=RunType.PERFORMANCE,
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            variant_label="default",
            status=RunStatus.COMPLETED,
            metrics={"episode_reward": 0.6, "train_timesteps": 50_000},
            artifact_refs=["manifest.json", "metrics.json"],
            started_at=datetime(2026, 4, 10, 12, 1, tzinfo=UTC),
            ended_at=datetime(2026, 4, 10, 12, 2, tzinfo=UTC),
        )
        return FakeExecutionResult(run=run)


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


def test_run_experiment_executes_parallel_candidate_batch() -> None:
    """Run tool should execute multiple candidates when candidate_ids are provided."""

    base_record = _record_from_fixture()
    spec = base_record.spec.model_copy(
        update={
            "budgets": base_record.spec.budgets.model_copy(
                update={
                    "compute": base_record.spec.budgets.compute.model_copy(
                        update={"max_parallel_experiments": 2}
                    )
                }
            )
        }
    )
    record = base_record.model_copy(update={"spec": spec})
    baseline = RewardCandidate(
        candidate_id=f"{record.experiment_id}-candidate-000",
        session_id=record.experiment_id,
        iteration_index=0,
        reward_definition="def reward(observation):\n    return 1.0\n",
        change_summary="baseline",
    )
    candidate_001 = RewardCandidate(
        candidate_id=f"{record.experiment_id}-candidate-001",
        session_id=record.experiment_id,
        parent_candidate_id=baseline.candidate_id,
        iteration_index=1,
        reward_definition="def reward(observation):\n    return 1.1\n",
        change_summary="sample one",
    )
    candidate_002 = RewardCandidate(
        candidate_id=f"{record.experiment_id}-candidate-002",
        session_id=record.experiment_id,
        parent_candidate_id=baseline.candidate_id,
        iteration_index=1,
        reward_definition="def reward(observation):\n    return 1.2\n",
        change_summary="sample two",
    )
    tool = RunExperimentTool(execution_service=FakeExecutionService())  # type: ignore[arg-type]

    result = tool.execute(
        record=record,
        candidates=[baseline, candidate_001, candidate_002],
        action_input={"candidate_ids": [candidate_001.candidate_id, candidate_002.candidate_id]},
        run_count=0,
    )

    assert result.status == "ok"
    assert "parallel experiment runs" in result.summary
    runs = result.payload["runs"]
    candidates = result.payload["candidates"]
    assert isinstance(runs, list)
    assert isinstance(candidates, list)
    assert len(runs) == 2
    assert len(candidates) == 2
    assert runs[0]["run_id"].endswith("-run-001")
    assert runs[1]["run_id"].endswith("-run-002")


def test_run_robustness_probes_returns_assessment_payload() -> None:
    """Robustness tool should return probe runs and risk assessment payload."""

    base_record = _record_from_fixture()
    runtime_dir = ".agentic-test-runtime-robust-tool"
    spec = base_record.spec.model_copy(
        update={
            "outputs": base_record.spec.outputs.model_copy(update={"runtime_dir": runtime_dir})
        }
    )
    record = base_record.model_copy(update={"spec": spec})
    candidate = RewardCandidate(
        candidate_id="experiment-tool-tests-candidate-000",
        session_id=record.experiment_id,
        iteration_index=0,
        reward_definition="def compute_reward(observation):\n    return 1.0\n",
        change_summary="baseline",
        aggregate_score=0.9,
    )
    performance_run = ExperimentRun(
        run_id="experiment-tool-tests-run-001",
        candidate_id=candidate.candidate_id,
        backend=record.spec.environment.backend,
        environment_id=record.spec.environment.id,
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        status=RunStatus.COMPLETED,
        metrics={"episode_reward": 0.9},
        artifact_refs=["manifest.json", "metrics.json"],
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
    )
    tool = RunRobustnessProbesTool(
        execution_service=ExperimentExecutionService(
            artifact_writer=RunArtifactWriter(Path(runtime_dir) / "runs")
        ),
        robustness_runner_factory=lambda *_: FakeRobustnessRunner(),
    )

    result = tool.execute(
        record=record,
        candidates=[candidate],
        runs=[performance_run],
        action_input={"candidate_id": candidate.candidate_id},
    )

    assert result.status == "ok"
    assert result.payload["candidate_id"] == candidate.candidate_id
    assert result.payload["assessment"]["risk_level"] == "high"
    assert len(result.payload["robustness_runs"]) == 1
