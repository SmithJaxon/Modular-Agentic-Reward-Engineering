"""
Summary: Unit tests for real-execution reward loading, artifacts, and persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import (
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
    ExperimentExecutionService,
)
from rewardlab.experiments.reward_program import RewardProgramStatus, load_reward_program
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend


@pytest.fixture()
def workspace_tmp_path() -> Path:
    """Provide a worktree-local temporary directory without pytest tmpdir helpers."""

    root = Path(".tmp") / f"real-foundations-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def build_candidate(**overrides: object) -> RewardCandidate:
    """Create a valid reward candidate for focused real-execution tests."""

    payload: dict[str, object] = {
        "candidate_id": "candidate-001",
        "session_id": "session-001",
        "iteration_index": 1,
        "reward_definition": (
            "def compute_reward(cart_position, pole_angle_radians, angular_velocity, terminated):\n"
            "    if terminated:\n"
            "        return -10.0\n"
            "    return 1.5 - abs(cart_position) - abs(pole_angle_radians)\n"
        ),
        "change_summary": "Initial real backend candidate.",
        "aggregate_score": 1.25,
    }
    payload.update(overrides)
    return RewardCandidate.model_validate(payload)


def test_load_reward_program_accepts_compute_reward_fallback() -> None:
    """The loader should accept the checked-in compute_reward entrypoint shape."""

    program = load_reward_program(
        candidate_id="candidate-001",
        source_text=build_candidate().reward_definition,
    )

    reward_callable = program.require_callable()

    assert program.validation_status == RewardProgramStatus.VALID
    assert program.entrypoint_name == "compute_reward"
    assert reward_callable(
        cart_position=0.1,
        pole_angle_radians=0.0,
        angular_velocity=0.2,
        terminated=False,
    ) == pytest.approx(1.4)


def test_load_reward_program_reports_actionable_error_for_missing_entrypoint() -> None:
    """The loader should expose why a candidate cannot be executed."""

    program = load_reward_program(
        candidate_id="candidate-001",
        source_text="value = 7\n",
    )

    assert program.validation_status == RewardProgramStatus.INVALID
    assert program.validation_error is not None
    assert "callable" in program.validation_error
    with pytest.raises(ValueError):
        program.require_callable()


def test_run_artifact_writer_emits_manifest_metrics_and_trace(workspace_tmp_path: Path) -> None:
    """A real run should produce metrics-first artifact files with a manifest."""

    writer = RunArtifactWriter(workspace_tmp_path / "artifacts")
    bundle = writer.write_bundle(
        run_id="run-001",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="CartPole-v1",
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        status=RunStatus.COMPLETED,
        metrics={"episode_reward": 12.5, "step_count": 4},
        event_trace=[{"step": 1, "reward": 1.0}, {"step": 2, "reward": 2.0}],
        manifest_metadata={"seed": 7},
    )

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(bundle.metrics_path.read_text(encoding="utf-8"))

    assert bundle.manifest_path.exists()
    assert bundle.metrics_path.exists()
    assert bundle.event_trace_path is not None
    assert bundle.event_trace_path.exists()
    assert manifest["run_id"] == "run-001"
    assert manifest["backend"] == "gymnasium"
    assert manifest["execution_mode"] == "actual_backend"
    assert manifest["files"]["metrics"] == "metrics.json"
    assert manifest["files"]["event_trace"] == "event_trace.json"
    assert manifest["metadata"]["seed"] == 7
    assert metrics["episode_reward"] == 12.5


def test_experiment_run_requires_artifact_refs_for_successful_actual_backend_runs() -> None:
    """Completed actual-backend runs should always point at persisted evidence."""

    started_at = datetime(2026, 4, 2, 22, 0, tzinfo=UTC)
    ended_at = datetime(2026, 4, 2, 22, 1, tzinfo=UTC)

    with pytest.raises(ValidationError):
        ExperimentRun(
            run_id="run-001",
            candidate_id="candidate-001",
            backend=EnvironmentBackend.GYMNASIUM,
            environment_id="CartPole-v1",
            run_type=RunType.PERFORMANCE,
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            variant_label="default",
            status=RunStatus.COMPLETED,
            metrics={"episode_reward": 12.5},
            started_at=started_at,
            ended_at=ended_at,
        )


def test_session_repository_round_trips_experiment_runs_and_assessments(
    workspace_tmp_path: Path,
) -> None:
    """The repository should persist experiment runs and robustness assessments."""

    repository = SessionRepository(
        RepositoryPaths(
            database_path=workspace_tmp_path / "rewardlab.sqlite3",
            event_log_path=workspace_tmp_path / "events" / "events.jsonl",
        )
    )
    repository.initialize()

    started_at = datetime(2026, 4, 2, 22, 0, tzinfo=UTC)
    ended_at = datetime(2026, 4, 2, 22, 1, tzinfo=UTC)
    run = ExperimentRun(
        run_id="run-001",
        candidate_id="candidate-001",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="CartPole-v1",
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        status=RunStatus.COMPLETED,
        metrics={"episode_reward": 12.5},
        artifact_refs=["artifacts/run-001/manifest.json"],
        started_at=started_at,
        ended_at=ended_at,
    )
    assessment = RobustnessAssessment(
        assessment_id="candidate-001-robustness",
        candidate_id="candidate-001",
        variant_count=3,
        degradation_ratio=0.25,
        risk_level=RiskLevel.MEDIUM,
        risk_notes="Worst probe run dropped by 25 percent.",
    )

    repository.save_experiment_run(run)
    repository.save_robustness_assessment(assessment)

    loaded_run = repository.get_experiment_run("run-001")
    loaded_assessment = repository.get_robustness_assessment("candidate-001-robustness")

    assert loaded_run is not None
    assert loaded_run.backend == EnvironmentBackend.GYMNASIUM
    assert loaded_run.execution_mode == ExecutionMode.ACTUAL_BACKEND
    assert repository.list_experiment_runs(candidate_id="candidate-001") == [loaded_run]
    assert loaded_assessment is not None
    assert repository.list_robustness_assessments(candidate_id="candidate-001") == [
        loaded_assessment
    ]


def test_execution_service_returns_completed_run_with_artifact_refs(
    workspace_tmp_path: Path,
) -> None:
    """Execution coordination should load the reward, write artifacts, and return a run."""

    service = ExperimentExecutionService(RunArtifactWriter(workspace_tmp_path / "artifacts"))
    candidate = build_candidate()
    request = ExecutionRequest(
        run_id="run-123",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="CartPole-v1",
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        seed=7,
    )

    def runner(
        execution_request: ExecutionRequest,
        reward_program,
    ) -> ExecutionOutcome:
        """Return a stable fake runtime outcome for the service test."""

        reward_value = reward_program.require_callable()(
            cart_position=0.1,
            pole_angle_radians=0.0,
            angular_velocity=0.0,
            terminated=False,
        )
        assert execution_request.run_id == "run-123"
        return ExecutionOutcome(
            metrics={"episode_reward": reward_value, "step_count": 1},
            event_trace=[{"step": 1, "reward": reward_value}],
            runtime_status=BackendRuntimeStatus(
                backend=EnvironmentBackend.GYMNASIUM,
                ready=True,
                status_reason="gymnasium import and environment resolution succeeded",
                detected_version="0.29.1",
            ),
        )

    result = service.execute_candidate(candidate=candidate, request=request, runner=runner)

    assert result.run.status == RunStatus.COMPLETED
    assert result.run.execution_mode == ExecutionMode.ACTUAL_BACKEND
    assert result.artifact_bundle is not None
    assert result.artifact_bundle.manifest_path.exists()
    assert any(path.endswith("manifest.json") for path in result.run.artifact_refs)
    assert result.runtime_status is not None
    assert result.runtime_status.ready is True


def test_execution_service_returns_failed_run_for_invalid_reward_program(
    workspace_tmp_path: Path,
) -> None:
    """Execution coordination should fail fast when candidate reward code is invalid."""

    service = ExperimentExecutionService(RunArtifactWriter(workspace_tmp_path / "artifacts"))
    candidate = build_candidate(
        reward_definition="def not_the_expected_name():\n    return 1.0\n"
    )
    request = ExecutionRequest(
        run_id="run-456",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="CartPole-v1",
    )

    def runner(
        execution_request: ExecutionRequest,
        reward_program,
    ) -> ExecutionOutcome:
        """Fail the test if execution continues past reward validation."""

        del execution_request, reward_program
        raise AssertionError("runner should not execute for invalid reward programs")

    result = service.execute_candidate(candidate=candidate, request=request, runner=runner)

    assert result.run.status == RunStatus.FAILED
    assert result.run.failure_reason is not None
    assert "callable" in result.run.failure_reason
    assert result.artifact_bundle is None


def test_execution_service_captures_runtime_failure_reason(workspace_tmp_path: Path) -> None:
    """Execution failures should be surfaced in the returned failed run record."""

    service = ExperimentExecutionService(RunArtifactWriter(workspace_tmp_path / "artifacts"))
    candidate = build_candidate()
    request = ExecutionRequest(
        run_id="run-789",
        backend=EnvironmentBackend.ISAACGYM,
        environment_id="isaac-task",
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )

    def runner(
        execution_request: ExecutionRequest,
        reward_program,
    ) -> ExecutionOutcome:
        """Raise a typed execution failure with runtime readiness details."""

        del execution_request, reward_program
        raise ExecutionError(
            "isaac runtime missing",
            runtime_status=BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason="isaacgym module import failed",
                missing_prerequisites=["install approved isaac runtime in .venv"],
            ),
        )

    result = service.execute_candidate(candidate=candidate, request=request, runner=runner)

    assert result.run.status == RunStatus.FAILED
    assert result.runtime_status is not None
    assert result.runtime_status.ready is False
    assert result.run.failure_reason is not None
    assert "isaac runtime missing" in result.run.failure_reason
