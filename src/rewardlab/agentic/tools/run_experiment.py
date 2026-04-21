"""
Summary: Worker tool that executes one candidate experiment run via configured backend.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rewardlab.agentic.contracts import ToolResult
from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import (
    ExecutionRequest,
    ExperimentExecutionService,
    ExperimentRunner,
)
from rewardlab.experiments.runner_factory import build_runner
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class RunExperimentTool:
    """Execute a candidate on the real runtime backend and return metrics."""

    def __init__(
        self,
        *,
        execution_service: ExperimentExecutionService,
    ) -> None:
        """Store execution dependencies."""

        self.execution_service = execution_service

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        action_input: dict[str, object],
        run_count: int,
    ) -> ToolResult:
        """Run one experiment and return a normalized tool result payload."""

        selected_candidates = _select_candidates(candidates=candidates, action_input=action_input)
        if len(selected_candidates) == 0:
            return ToolResult(
                status="error",
                summary="no candidate selected for experiment execution",
                payload={},
            )
        spec = record.spec
        rollout = spec.execution.rollout
        runner = build_runner(
            environment_backend=spec.environment.backend,
            ppo_config=spec.execution.ppo,
            isaac_config=spec.execution.isaac,
        )
        execution_service = _resolve_execution_service(
            default_service=self.execution_service,
            runtime_dir=Path(spec.outputs.runtime_dir),
        )
        max_parallel = max(spec.budgets.compute.max_parallel_experiments, 1)
        dispatch_candidates = selected_candidates[:max_parallel]
        if len(dispatch_candidates) == 1:
            run = _execute_candidate_run(
                execution_service=execution_service,
                runner=runner,
                candidate=dispatch_candidates[0],
                request=_build_request(
                    record=record,
                    run_id=f"{record.experiment_id}-run-{run_count + 1:03d}",
                    max_episode_steps=rollout.max_episode_steps if rollout is not None else None,
                ),
            )
            if run.status.value != "completed":
                return ToolResult(
                    status="error",
                    summary=run.failure_reason or "experiment run failed",
                    payload={"run": run.model_dump(mode="json")},
                )
            score = _score_from_run(run.metrics)
            updated_candidate = dispatch_candidates[0].model_copy(update={"aggregate_score": score})
            return ToolResult(
                status="ok",
                summary=(
                    f"Executed {run.run_id} for {dispatch_candidates[0].candidate_id}; "
                    f"score={score:.6f}."
                ),
                payload={
                    "run": run.model_dump(mode="json"),
                    "candidate": updated_candidate.model_dump(mode="json"),
                },
            )

        run_plan = [
            (
                index,
                candidate,
                _build_request(
                    record=record,
                    run_id=f"{record.experiment_id}-run-{run_count + index + 1:03d}",
                    max_episode_steps=rollout.max_episode_steps if rollout is not None else None,
                ),
            )
            for index, candidate in enumerate(dispatch_candidates)
        ]
        completed_entries: list[tuple[int, dict[str, object], dict[str, object]]] = []
        failed_entries: list[tuple[int, dict[str, object]]] = []
        with ThreadPoolExecutor(
            max_workers=len(run_plan),
            thread_name_prefix="rewardlab-run",
        ) as executor:
            futures = {
                executor.submit(
                    _execute_candidate_run,
                    execution_service=execution_service,
                    runner=runner,
                    candidate=candidate,
                    request=request,
                ): (index, candidate)
                for index, candidate, request in run_plan
            }
            for future in as_completed(futures):
                index, candidate = futures[future]
                try:
                    run = future.result()
                except Exception as exc:
                    failed_entries.append(
                        (
                            index,
                            {
                                "candidate_id": candidate.candidate_id,
                                "failure_reason": f"experiment execution raised: {exc}",
                            },
                        )
                    )
                    continue

                run_payload = run.model_dump(mode="json")
                if run.status.value == "completed":
                    score = _score_from_run(run.metrics)
                    updated_candidate = candidate.model_copy(update={"aggregate_score": score})
                    completed_entries.append(
                        (
                            index,
                            run_payload,
                            updated_candidate.model_dump(mode="json"),
                        )
                    )
                else:
                    failed_entries.append((index, run_payload))

        completed_entries.sort(key=lambda item: item[0])
        failed_entries.sort(key=lambda item: item[0])
        payload: dict[str, object] = {
            "runs": [run_payload for _, run_payload, _ in completed_entries],
            "candidates": [candidate_payload for _, _, candidate_payload in completed_entries],
            "candidate_ids": [candidate.candidate_id for candidate in dispatch_candidates],
        }
        if len(failed_entries) > 0:
            payload["failed_runs"] = [run_payload for _, run_payload in failed_entries]

        completed_count = len(completed_entries)
        failed_count = len(failed_entries)
        if completed_count == 0:
            summary = "all parallel experiment runs failed"
            if failed_count > 0:
                first_failure = failed_entries[0][1]
                if isinstance(first_failure, dict):
                    failure_reason = first_failure.get("failure_reason")
                    if isinstance(failure_reason, str) and failure_reason.strip():
                        summary = failure_reason
            return ToolResult(status="error", summary=summary, payload=payload)

        summary = (
            f"Executed {completed_count} parallel experiment runs for "
            f"{len(dispatch_candidates)} candidates."
        )
        if failed_count > 0:
            summary = f"{summary} Failed runs: {failed_count}."
        return ToolResult(status="ok", summary=summary, payload=payload)


def _select_candidates(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> list[RewardCandidate]:
    """Return one or more target candidates for execution."""

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[RewardCandidate] = []
    requested_ids = action_input.get("candidate_ids")
    if isinstance(requested_ids, list):
        for value in requested_ids:
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            candidate = candidate_by_id.get(cleaned)
            if candidate is not None and candidate not in selected:
                selected.append(candidate)

    requested_id = action_input.get("candidate_id")
    if isinstance(requested_id, str):
        cleaned = requested_id.strip()
        candidate = candidate_by_id.get(cleaned)
        if candidate is not None and candidate not in selected:
            selected.insert(0, candidate)

    if len(selected) > 0:
        return selected
    if len(candidates) == 0:
        return []
    return [max(candidates, key=lambda candidate: candidate.iteration_index)]


def _build_request(
    *,
    record: AgentExperimentRecord,
    run_id: str,
    max_episode_steps: int | None,
) -> ExecutionRequest:
    """Build one backend execution request for a candidate run."""

    return ExecutionRequest(
        run_id=run_id,
        backend=record.spec.environment.backend,
        environment_id=record.spec.environment.id,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        seed=record.spec.environment.seed,
        entrypoint_name=record.spec.baseline_reward.entrypoint_name,
        max_episode_steps=max_episode_steps,
    )


def _execute_candidate_run(
    *,
    execution_service: ExperimentExecutionService,
    runner: ExperimentRunner,
    candidate: RewardCandidate,
    request: ExecutionRequest,
) -> ExperimentRun:
    """Execute one candidate and return its normalized run object."""

    result = execution_service.execute_candidate(
        candidate=candidate,
        request=request,
        runner=runner,
    )
    return result.run


def _score_from_run(metrics: dict[str, object]) -> float:
    """Extract the canonical score value from run metrics."""

    episode_reward = metrics.get("episode_reward")
    if isinstance(episode_reward, (int, float)):
        return float(episode_reward)
    total_reward = metrics.get("total_reward")
    if isinstance(total_reward, (int, float)):
        return float(total_reward)
    return 0.0


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
