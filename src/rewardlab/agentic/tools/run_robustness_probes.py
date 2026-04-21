"""
Summary: Worker tool that runs robustness probes for reward-hacking risk checks.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from rewardlab.agentic.contracts import ToolResult
from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import ExperimentExecutionService, ExperimentRunner
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.experiments.runner_factory import build_runner
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import EnvironmentBackend


class RobustnessRunnerLike(Protocol):
    """Minimal runner interface required by the robustness worker tool."""

    def run_candidate_probes(
        self,
        *,
        candidate: RewardCandidate,
        primary_run: ExperimentRun,
        environment_backend: EnvironmentBackend,
        environment_id: str,
        policy: Callable[[Any], Any] | None = None,
    ) -> tuple[list[ExperimentRun], RobustnessAssessment | None]:
        """Execute robustness probes and return runs plus optional assessment."""


RobustnessRunnerFactory = Callable[
    [Path, ExperimentExecutionService, ExperimentRunner],
    RobustnessRunnerLike,
]


class RunRobustnessProbesTool:
    """Execute robustness probes for one candidate via the shared robustness runner."""

    def __init__(
        self,
        *,
        execution_service: ExperimentExecutionService,
        robustness_runner_factory: RobustnessRunnerFactory | None = None,
    ) -> None:
        """Store execution dependencies for probe-run orchestration."""

        self.execution_service = execution_service
        self.robustness_runner_factory = (
            robustness_runner_factory or _build_robustness_runner
        )

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Run robustness probes for one candidate and return normalized payloads."""

        selected_candidate = _select_candidate(candidates=candidates, action_input=action_input)
        primary_run = _select_primary_run(
            candidate_id=selected_candidate.candidate_id,
            runs=runs,
            action_input=action_input,
        )
        if primary_run is None:
            return ToolResult(
                status="error",
                summary=(
                    "No completed primary performance run is available for the candidate; "
                    "robustness probes require one baseline run."
                ),
                payload={"candidate_id": selected_candidate.candidate_id},
            )

        spec = record.spec
        runtime_dir = Path(spec.outputs.runtime_dir)
        probe_matrix_path = _write_probe_matrix(
            runtime_dir=runtime_dir,
            record=record,
            candidate=selected_candidate,
            primary_run=primary_run,
            action_input=action_input,
        )
        execution_service = _resolve_execution_service(
            default_service=self.execution_service,
            runtime_dir=runtime_dir,
        )
        runner = build_runner(
            environment_backend=spec.environment.backend,
            ppo_config=spec.execution.ppo,
            isaac_config=spec.execution.isaac,
        )
        robustness_runner = self.robustness_runner_factory(
            probe_matrix_path,
            execution_service,
            runner,
        )
        probe_runs, assessment = robustness_runner.run_candidate_probes(
            candidate=selected_candidate,
            primary_run=primary_run,
            environment_backend=spec.environment.backend,
            environment_id=spec.environment.id,
        )
        payload: dict[str, object] = {
            "candidate_id": selected_candidate.candidate_id,
            "primary_run_id": primary_run.run_id,
            "probe_matrix_path": str(probe_matrix_path),
            "robustness_runs": [run.model_dump(mode="json") for run in probe_runs],
            "assessment": (
                assessment.model_dump(mode="json") if assessment is not None else None
            ),
        }
        if assessment is None:
            return ToolResult(
                status="error",
                summary=(
                    "Robustness probes completed without a valid robustness assessment."
                ),
                payload=payload,
            )

        failed_runs = [
            run for run in probe_runs if run.status != RunStatus.COMPLETED
        ]
        summary = (
            f"Executed {len(probe_runs)} robustness probes for {selected_candidate.candidate_id}; "
            f"risk={assessment.risk_level.value}, degradation={assessment.degradation_ratio:.4f}."
        )
        if len(failed_runs) > 0:
            summary = f"{summary} Failed probes: {len(failed_runs)}."

        return ToolResult(
            status="ok",
            summary=summary,
            payload=payload,
        )


def _select_candidate(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> RewardCandidate:
    """Return the target candidate for robustness probing."""

    requested_id = action_input.get("candidate_id")
    if isinstance(requested_id, str):
        for candidate in candidates:
            if candidate.candidate_id == requested_id:
                return candidate
    scored = [candidate for candidate in candidates if candidate.aggregate_score is not None]
    if len(scored) > 0:
        return max(scored, key=lambda candidate: candidate.aggregate_score or float("-inf"))
    return max(candidates, key=lambda candidate: candidate.iteration_index)


def _select_primary_run(
    *,
    candidate_id: str,
    runs: list[ExperimentRun],
    action_input: dict[str, object],
) -> ExperimentRun | None:
    """Return the baseline performance run used for robustness degradation scoring."""

    requested_run_id = action_input.get("primary_run_id")
    if isinstance(requested_run_id, str):
        for run in runs:
            if (
                run.run_id == requested_run_id
                and run.candidate_id == candidate_id
                and run.run_type == RunType.PERFORMANCE
                and run.status == RunStatus.COMPLETED
            ):
                return run
    matching = [
        run
        for run in runs
        if run.candidate_id == candidate_id
        and run.run_type == RunType.PERFORMANCE
        and run.status == RunStatus.COMPLETED
    ]
    if len(matching) == 0:
        return None
    return matching[-1]


def _write_probe_matrix(
    *,
    runtime_dir: Path,
    record: AgentExperimentRecord,
    candidate: RewardCandidate,
    primary_run: ExperimentRun,
    action_input: dict[str, object],
) -> Path:
    """Write a run-scoped probe matrix used by the robustness runner."""

    probe_variants = _probe_variants(
        record=record,
        primary_run=primary_run,
        action_input=action_input,
    )
    document = {
        "version": 1,
        "description": "Agentic robustness probes generated for autonomous loop execution.",
        "backends": {
            record.spec.environment.backend.value: probe_variants,
        },
    }
    safe_candidate = candidate.candidate_id.replace(":", "-")
    matrix_dir = runtime_dir / "robustness" / "probe_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{record.experiment_id}-{safe_candidate}.json"
    matrix_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return matrix_path


def _probe_variants(
    *,
    record: AgentExperimentRecord,
    primary_run: ExperimentRun,
    action_input: dict[str, object],
) -> list[dict[str, object]]:
    """Return explicit probe variants from input or generated seed sweeps."""

    requested = action_input.get("probe_variants")
    if isinstance(requested, list):
        parsed = [_parse_variant(item) for item in requested if isinstance(item, dict)]
        if len(parsed) > 0:
            return parsed

    base_seed = record.spec.environment.seed
    requested_seeds = action_input.get("probe_seeds")
    if isinstance(requested_seeds, list):
        parsed_seeds = [int(item) for item in requested_seeds if isinstance(item, int)]
    else:
        parsed_seeds = []
    if len(parsed_seeds) == 0:
        parsed_seeds = [101, 202] if base_seed is None else [base_seed + 11, base_seed + 23]

    environment_id = primary_run.environment_id
    variants: list[dict[str, object]] = []
    for index, seed in enumerate(parsed_seeds, start=1):
        variants.append(
            {
                "label": f"{environment_id.lower()}-seed-{index}",
                "environment_id": environment_id,
                "seed": seed,
                "overrides": {"probe_seed": seed},
            }
        )
    return variants


def _parse_variant(raw: dict[str, object]) -> dict[str, object]:
    """Normalize one user-provided probe variant definition."""

    label_value = raw.get("label")
    label = str(label_value).strip() if isinstance(label_value, str) else "probe-variant"
    environment_value = raw.get("environment_id")
    environment_id = (
        str(environment_value).strip()
        if isinstance(environment_value, str) and environment_value.strip()
        else "CartPole-v1"
    )
    seed_value = raw.get("seed")
    seed = int(seed_value) if isinstance(seed_value, int) else 0
    overrides_value = raw.get("overrides")
    overrides: dict[str, float | int]
    if isinstance(overrides_value, dict):
        overrides = {
            key: value
            for key, value in overrides_value.items()
            if isinstance(key, str) and isinstance(value, (int, float))
        }
    else:
        overrides = {"probe_seed": seed}
    return {
        "label": label,
        "environment_id": environment_id,
        "seed": seed,
        "overrides": overrides,
    }


def _resolve_execution_service(
    *,
    default_service: ExperimentExecutionService,
    runtime_dir: Path,
) -> ExperimentExecutionService:
    """Return an execution service scoped to the runtime configured by the spec."""

    requested_root = runtime_dir / "runs"
    current_root = default_service.artifact_writer.root_dir
    if current_root == requested_root:
        return default_service
    return ExperimentExecutionService(artifact_writer=RunArtifactWriter(requested_root))


def _build_robustness_runner(
    probe_matrix_path: Path,
    execution_service: ExperimentExecutionService,
    runner: ExperimentRunner,
) -> RobustnessRunnerLike:
    """Build the default robustness runner for agentic probe execution."""

    return RobustnessRunner(
        probe_matrix_path=probe_matrix_path,
        experiment_execution_service=execution_service,
        gymnasium_runner=runner,
        isaacgym_runner=runner,
    )
