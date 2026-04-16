"""
Summary: Worker tool that executes one candidate experiment run via Gymnasium.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from pathlib import Path

from rewardlab.agentic.contracts import ToolResult
from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import ExecutionRequest, ExperimentExecutionService
from rewardlab.experiments.gymnasium_runner import (
    GymnasiumExperimentRunner,
    HumanoidPpoEvaluationConfig,
)
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExecutionMode
from rewardlab.schemas.reward_candidate import RewardCandidate


class RunExperimentTool:
    """Execute a candidate on the real Gymnasium backend and return metrics."""

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

        selected_candidate = _select_candidate(candidates=candidates, action_input=action_input)
        spec = record.spec
        ppo = spec.execution.ppo
        rollout = spec.execution.rollout
        runner = GymnasiumExperimentRunner(
            humanoid_ppo_config=(
                HumanoidPpoEvaluationConfig(
                    total_timesteps=ppo.total_timesteps,
                    checkpoint_count=ppo.checkpoint_count,
                    evaluation_run_count=ppo.eval_runs,
                    evaluation_episodes_per_checkpoint=ppo.eval_episodes_per_checkpoint,
                )
                if ppo is not None
                else None
            )
        )
        run_id = f"{record.experiment_id}-run-{run_count + 1:03d}"
        request = ExecutionRequest(
            run_id=run_id,
            backend=spec.environment.backend,
            environment_id=spec.environment.id,
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            seed=spec.environment.seed,
            entrypoint_name=spec.baseline_reward.entrypoint_name,
            max_episode_steps=rollout.max_episode_steps if rollout is not None else None,
        )
        execution_service = _resolve_execution_service(
            default_service=self.execution_service,
            runtime_dir=Path(spec.outputs.runtime_dir),
        )
        result = execution_service.execute_candidate(
            candidate=selected_candidate,
            request=request,
            runner=runner,
        )
        run = result.run
        if run.status.value != "completed":
            return ToolResult(
                status="error",
                summary=run.failure_reason or "experiment run failed",
                payload={"run": run.model_dump(mode="json")},
            )
        score = _score_from_run(run.metrics)
        updated_candidate = selected_candidate.model_copy(update={"aggregate_score": score})
        return ToolResult(
            status="ok",
            summary=(
                f"Executed {run.run_id} for {selected_candidate.candidate_id}; "
                f"score={score:.6f}."
            ),
            payload={
                "run": run.model_dump(mode="json"),
                "candidate": updated_candidate.model_dump(mode="json"),
            },
        )


def _select_candidate(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> RewardCandidate:
    """Return the target candidate for execution."""

    requested_id = action_input.get("candidate_id")
    if isinstance(requested_id, str):
        for candidate in candidates:
            if candidate.candidate_id == requested_id:
                return candidate
    return max(candidates, key=lambda candidate: candidate.iteration_index)


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
