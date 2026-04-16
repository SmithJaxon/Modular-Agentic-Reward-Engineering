"""
Summary: Unit tests for agentic tool executors and budget-aware tool gating.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.experiments.backends.base import ExperimentOutput
from rewardlab.schemas.budget_state import BudgetState, BudgetUsage
from rewardlab.schemas.experiment_run import (
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentRunType,
)
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment
from rewardlab.schemas.tool_contracts import ToolRequest, ToolResultStatus
from rewardlab.selection.risk_analyzer import RiskAnalysisResult
from rewardlab.tools.broker import ToolBroker, build_default_registry
from rewardlab.tools.executors import (
    compare_candidates_executor,
    export_report_executor,
    run_experiment_executor,
    run_probe_suite_executor,
)


class _FakeAdapter:
    """
    Return deterministic performance and reflection outputs for executor tests.
    """

    def run_performance(self, payload: object) -> ExperimentOutput:
        """
        Return a fixed performance payload.
        """
        _ = payload
        return ExperimentOutput(
            score=123.4,
            metrics={
                "total_timesteps": 2048,
                "evaluation_episodes_consumed": 7,
                "score": 123.4,
            },
            summary="fake performance summary",
            artifact_refs=("artifact/performance.json",),
        )

    def run_reflection(self, payload: object) -> ExperimentOutput:
        """
        Return a fixed reflection payload.
        """
        _ = payload
        return ExperimentOutput(
            score=123.4,
            metrics={"score": 123.4},
            summary="fake reflection summary",
            artifact_refs=("artifact/reflection.txt",),
        )


class _FakeRobustnessRunner:
    """
    Return deterministic robustness output for probe-suite executor tests.
    """

    def run(self, candidate_id: str, payload: object, primary_score: float) -> object:
        """
        Return fixed experiment runs and risk-analysis payload.
        """
        _ = payload
        _ = primary_score
        run = ExperimentRun(
            run_id="run-probe",
            candidate_id=candidate_id,
            run_type=ExperimentRunType.ROBUSTNESS,
            variant_label="observation_dropout",
            seed=13,
            status=ExperimentRunStatus.COMPLETED,
            metrics={"score": 100.0, "total_timesteps": 128, "evaluation_episodes_consumed": 1},
            artifact_refs=["artifact/probe.json"],
            started_at="2026-04-10T00:00:00+00:00",
            ended_at="2026-04-10T00:00:00+00:00",
        )
        analysis = RiskAnalysisResult(
            assessment=RobustnessAssessment(
                assessment_id="assess-test",
                candidate_id=candidate_id,
                variant_count=1,
                degradation_ratio=0.1,
                risk_level=RiskLevel.MEDIUM,
                risk_notes="moderate test risk",
                created_at="2026-04-10T00:00:00+00:00",
            ),
            robustness_bonus=-0.03,
            tradeoff_rationale="test rationale",
            minor_robustness_risk_accepted=True,
        )
        return type(
            "_Result",
            (),
            {"experiment_runs": [run], "analysis": analysis},
        )()


def _budget_state() -> BudgetState:
    """
    Build a permissive budget state for executor tests.
    """
    return BudgetState(
        max_wall_clock_minutes=10,
        max_training_timesteps=100000,
        max_evaluation_episodes=100,
        max_api_input_tokens=10000,
        max_api_output_tokens=10000,
        max_api_usd=10.0,
        max_calls_per_model={"gpt-5.4-mini": 10},
    )


@pytest.mark.unit
def test_run_experiment_executor_resolves_files_and_returns_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify run_experiment executor validates arguments and returns normalized output.
    """
    objective_file = Path(".tmp-agentic-objective.txt")
    reward_file = Path(".tmp-agentic-reward.py")
    objective_file.write_text("maximize stability", encoding="utf-8")
    reward_file.write_text("def compute_reward(*args): return 1.0, {}", encoding="utf-8")

    monkeypatch.setattr(
        "rewardlab.tools.executors.resolve_backend_adapter",
        lambda _: _FakeAdapter(),
    )

    request = ToolRequest(
        request_id="toolreq-test",
        turn_index=0,
        tool_name="run_experiment",
        arguments={
            "session_id": "agentrun-test",
            "environment_id": "Humanoid-v4",
            "environment_backend": "gymnasium",
            "objective_file": str(objective_file),
            "reward_file": str(reward_file),
            "iteration_index": 2,
            "seed": 11,
            "variant_label": "default",
            "include_reflection": True,
            "overrides": {"execution_mode": "deterministic", "llm_model": "gpt-5.4-mini"},
        },
        rationale="unit test call",
        requested_at="2026-04-10T00:00:00+00:00",
    )

    result = run_experiment_executor(request, _budget_state())
    assert result.status is ToolResultStatus.COMPLETED
    assert result.output["score"] == 123.4
    assert result.output["reflection_summary"] == "fake reflection summary"
    assert result.training_timesteps == 2048
    assert result.evaluation_episodes == 7
    assert set(result.artifact_refs) == {"artifact/performance.json", "artifact/reflection.txt"}
    assert result.model_used == "gpt-5.4-mini"


@pytest.mark.unit
def test_run_experiment_executor_rejects_missing_inputs() -> None:
    """
    Verify missing required arguments produce a rejected tool result.
    """
    request = ToolRequest(
        request_id="toolreq-missing",
        turn_index=0,
        tool_name="run_experiment",
        arguments={"environment_backend": "gymnasium"},
        rationale="unit test missing args",
        requested_at="2026-04-10T00:00:00+00:00",
    )
    result = run_experiment_executor(request, _budget_state())
    assert result.status is ToolResultStatus.REJECTED
    assert result.error is not None


@pytest.mark.unit
def test_run_probe_suite_executor_returns_assessment_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify run_probe_suite executor returns normalized robustness outputs.
    """
    objective_file = Path(".tmp-agentic-objective-probe.txt")
    reward_file = Path(".tmp-agentic-reward-probe.py")
    objective_file.write_text("maximize stability", encoding="utf-8")
    reward_file.write_text("def compute_reward(*args): return 1.0, {}", encoding="utf-8")
    monkeypatch.setattr(
        "rewardlab.tools.executors.RobustnessRunner",
        lambda: _FakeRobustnessRunner(),
    )
    request = ToolRequest(
        request_id="toolreq-probe",
        turn_index=1,
        tool_name="run_probe_suite",
        arguments={
            "session_id": "agentrun-test",
            "candidate_id": "cand-test",
            "primary_score": 123.4,
            "environment_id": "Humanoid-v4",
            "environment_backend": "gymnasium",
            "objective_file": str(objective_file),
            "reward_file": str(reward_file),
            "iteration_index": 2,
            "seed": 11,
            "variant_label": "default",
            "overrides": {"llm_model": "gpt-5.4-mini"},
        },
        rationale="unit probe",
        requested_at="2026-04-10T00:00:00+00:00",
    )
    result = run_probe_suite_executor(request, _budget_state())
    assert result.status is ToolResultStatus.COMPLETED
    assert result.output["candidate_id"] == "cand-test"
    assert result.output["risk_level"] == "medium"
    assert result.output["robustness_bonus"] == -0.03
    assert result.training_timesteps == 128
    assert result.evaluation_episodes == 1


@pytest.mark.unit
def test_compare_candidates_executor_ranks_by_aggregate_score() -> None:
    """
    Verify compare_candidates uses score + robustness_bonus for ranking.
    """
    request = ToolRequest(
        request_id="toolreq-compare",
        turn_index=2,
        tool_name="compare_candidates",
        arguments={
            "candidates": [
                {"candidate_id": "cand-a", "score": 10.0, "robustness_bonus": -0.5},
                {"candidate_id": "cand-b", "score": 9.8, "robustness_bonus": 0.2},
            ]
        },
        rationale="unit compare",
        requested_at="2026-04-10T00:00:00+00:00",
    )
    result = compare_candidates_executor(request, _budget_state())
    assert result.status is ToolResultStatus.COMPLETED
    assert result.output["best_candidate_id"] == "cand-b"


@pytest.mark.unit
def test_export_report_executor_writes_artifact_file() -> None:
    """
    Verify export_report writes report JSON and returns an artifact reference.
    """
    output_path = Path(".tmp-agentic-export-report.json")
    if output_path.exists():
        output_path.unlink()
    request = ToolRequest(
        request_id="toolreq-export",
        turn_index=3,
        tool_name="export_report",
        arguments={
            "output_path": str(output_path),
            "report_payload": {"run_id": "agentrun-test", "best_score": 1.23},
        },
        rationale="unit export",
        requested_at="2026-04-10T00:00:00+00:00",
    )
    result = export_report_executor(request, _budget_state())
    assert result.status is ToolResultStatus.COMPLETED
    assert output_path.exists()
    assert result.artifact_refs == (str(output_path),)


@pytest.mark.unit
def test_budget_engine_allows_non_compute_tools_after_compute_budget_exhaustion() -> None:
    """
    Verify budget gating is tool-aware and does not over-restrict non-compute tools.
    """
    state = BudgetState(
        max_wall_clock_minutes=10,
        max_training_timesteps=100,
        max_evaluation_episodes=10,
        max_api_input_tokens=10000,
        max_api_output_tokens=10000,
        max_api_usd=10.0,
        max_calls_per_model={},
        usage=BudgetUsage(
            wall_clock_minutes=1.0,
            training_timesteps=100,
            evaluation_episodes=10,
        ),
    )
    engine = BudgetEngine(state, start_time=datetime.now(UTC))
    compute_allowed, _ = engine.can_execute(tool_name="run_experiment")
    read_allowed, _ = engine.can_execute(tool_name="read_artifact")
    assert compute_allowed is False
    assert read_allowed is True


@pytest.mark.unit
def test_tool_broker_rejects_oversized_run_experiment_request() -> None:
    """
    Verify broker pre-checks reject run_experiment calls that exceed remaining budgets.
    """
    state = BudgetState(
        max_wall_clock_minutes=10,
        max_training_timesteps=1000,
        max_evaluation_episodes=10,
        max_api_input_tokens=10000,
        max_api_output_tokens=10000,
        max_api_usd=10.0,
        max_calls_per_model={},
    )
    engine = BudgetEngine(state, start_time=datetime.now(UTC))
    broker = ToolBroker(
        registry=build_default_registry(),
        budget_engine=engine,
        enabled_tools=("run_experiment",),
    )
    result = broker.execute(
        turn_index=0,
        tool_name="run_experiment",
        arguments={
            "overrides": {
                "ppo_total_timesteps": 5000,
                "evaluation_episodes": 12,
            }
        },
        rationale="oversized test request",
    )
    assert result.status is ToolResultStatus.REJECTED
    assert result.error is not None
    assert "exceeds remaining" in result.error


@pytest.mark.unit
def test_tool_broker_estimates_probe_suite_multipliers_for_budget_check() -> None:
    """
    Verify run_probe_suite estimate accounts for multiple probe variants.
    """
    state = BudgetState(
        max_wall_clock_minutes=10,
        max_training_timesteps=10000,
        max_evaluation_episodes=14,
        max_api_input_tokens=10000,
        max_api_output_tokens=10000,
        max_api_usd=10.0,
        max_calls_per_model={},
    )
    engine = BudgetEngine(state, start_time=datetime.now(UTC))
    broker = ToolBroker(
        registry=build_default_registry(),
        budget_engine=engine,
        enabled_tools=("run_probe_suite",),
    )
    result = broker.execute(
        turn_index=0,
        tool_name="run_probe_suite",
        arguments={
            "overrides": {
                "ppo_total_timesteps": 1024,
                "evaluation_episodes": 5,
            }
        },
        rationale="probe multiplier budget check",
    )
    assert result.status is ToolResultStatus.REJECTED
    assert result.error is not None
    assert "evaluation episode budget" in result.error
