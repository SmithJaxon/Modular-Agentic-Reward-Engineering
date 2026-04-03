"""
Summary: Shared coordination layer for reward loading, execution, and artifact writing.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from rewardlab.experiments.artifacts import RunArtifactBundle, RunArtifactWriter
from rewardlab.experiments.reward_program import (
    DEFAULT_ENTRYPOINT_NAME,
    RewardProgram,
    RewardProgramStatus,
    load_reward_program_from_candidate,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend

__all__ = [
    "ExecutionError",
    "ExecutionOutcome",
    "ExecutionRequest",
    "ExecutionResult",
    "ExperimentExecutionService",
]


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Inputs describing one candidate experiment execution request."""

    run_id: str
    backend: EnvironmentBackend
    environment_id: str
    run_type: RunType = RunType.PERFORMANCE
    execution_mode: ExecutionMode = ExecutionMode.ACTUAL_BACKEND
    variant_label: str = "default"
    seed: int | None = None
    entrypoint_name: str = DEFAULT_ENTRYPOINT_NAME
    render_mode: str | None = None
    max_episode_steps: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Runtime outputs returned by a backend-specific experiment runner."""

    metrics: dict[str, Any]
    event_trace: list[dict[str, Any]] | None = None
    runtime_status: BackendRuntimeStatus | None = None
    manifest_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Structured result returned after execution coordination completes."""

    reward_program: RewardProgram
    run: ExperimentRun
    artifact_bundle: RunArtifactBundle | None = None
    runtime_status: BackendRuntimeStatus | None = None


class ExperimentRunner(Protocol):
    """Callable contract implemented by backend-specific experiment runners."""

    def __call__(
        self,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
    ) -> ExecutionOutcome:
        """Execute the request and return metrics, traces, and runtime status."""


class ExecutionError(RuntimeError):
    """Typed execution failure carrying optional backend readiness details."""

    def __init__(
        self,
        message: str,
        *,
        runtime_status: BackendRuntimeStatus | None = None,
    ) -> None:
        """Store failure details for later conversion into a failed run record."""

        super().__init__(message)
        self.runtime_status = runtime_status


class ExperimentExecutionService:
    """Coordinate reward-program loading, runner execution, and artifact writing."""

    def __init__(
        self,
        artifact_writer: RunArtifactWriter,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Store the shared artifact writer and injectable clock for tests."""

        self.artifact_writer = artifact_writer
        self.clock = clock or _utc_now

    def execute_candidate(
        self,
        *,
        candidate: RewardCandidate,
        request: ExecutionRequest,
        runner: ExperimentRunner,
    ) -> ExecutionResult:
        """Load a candidate reward program and execute it through the supplied runner."""

        reward_program = load_reward_program_from_candidate(
            candidate,
            entrypoint_name=request.entrypoint_name,
        )
        started_at = self.clock()
        if reward_program.validation_status != RewardProgramStatus.VALID:
            failed_run = self._build_failed_run(
                request=request,
                candidate_id=candidate.candidate_id,
                started_at=started_at,
                ended_at=self.clock(),
                failure_reason=reward_program.validation_error or "reward program is invalid",
            )
            return ExecutionResult(reward_program=reward_program, run=failed_run)

        try:
            outcome = runner(request, reward_program)
            artifact_bundle = self.artifact_writer.write_bundle(
                run_id=request.run_id,
                backend=request.backend,
                environment_id=request.environment_id,
                execution_mode=request.execution_mode,
                status=RunStatus.COMPLETED,
                metrics=outcome.metrics,
                event_trace=outcome.event_trace,
                manifest_metadata=outcome.manifest_metadata,
            )
            run = ExperimentRun(
                run_id=request.run_id,
                candidate_id=candidate.candidate_id,
                backend=request.backend,
                environment_id=request.environment_id,
                run_type=request.run_type,
                execution_mode=request.execution_mode,
                variant_label=request.variant_label,
                seed=request.seed,
                status=RunStatus.COMPLETED,
                metrics=outcome.metrics,
                artifact_refs=_artifact_refs_from_bundle(artifact_bundle),
                started_at=started_at,
                ended_at=self.clock(),
            )
            return ExecutionResult(
                reward_program=reward_program,
                run=run,
                artifact_bundle=artifact_bundle,
                runtime_status=outcome.runtime_status,
            )
        except ExecutionError as exc:
            failed_run = self._build_failed_run(
                request=request,
                candidate_id=candidate.candidate_id,
                started_at=started_at,
                ended_at=self.clock(),
                failure_reason=str(exc),
            )
            return ExecutionResult(
                reward_program=reward_program,
                run=failed_run,
                runtime_status=exc.runtime_status,
            )
        except Exception as exc:
            failed_run = self._build_failed_run(
                request=request,
                candidate_id=candidate.candidate_id,
                started_at=started_at,
                ended_at=self.clock(),
                failure_reason=f"experiment execution failed: {exc}",
            )
            return ExecutionResult(reward_program=reward_program, run=failed_run)

    def _build_failed_run(
        self,
        *,
        request: ExecutionRequest,
        candidate_id: str,
        started_at: datetime,
        ended_at: datetime,
        failure_reason: str,
    ) -> ExperimentRun:
        """Construct a failed experiment run for validation or execution errors."""

        return ExperimentRun(
            run_id=request.run_id,
            candidate_id=candidate_id,
            backend=request.backend,
            environment_id=request.environment_id,
            run_type=request.run_type,
            execution_mode=request.execution_mode,
            variant_label=request.variant_label,
            seed=request.seed,
            status=RunStatus.FAILED,
            metrics={},
            artifact_refs=[],
            failure_reason=failure_reason,
            started_at=started_at,
            ended_at=ended_at,
        )


def _artifact_refs_from_bundle(bundle: RunArtifactBundle) -> list[str]:
    """Return the stable artifact references recorded on a completed run."""

    artifact_refs = [str(bundle.manifest_path), str(bundle.metrics_path)]
    if bundle.event_trace_path is not None:
        artifact_refs.append(str(bundle.event_trace_path))
    if bundle.frame_dir is not None:
        artifact_refs.append(str(bundle.frame_dir))
    if bundle.video_path is not None:
        artifact_refs.append(str(bundle.video_path))
    return artifact_refs


def _utc_now() -> datetime:
    """Return the current UTC time for execution timestamps."""

    return datetime.now(UTC)
