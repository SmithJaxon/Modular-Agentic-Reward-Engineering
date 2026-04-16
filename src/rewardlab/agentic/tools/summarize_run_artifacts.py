"""
Summary: Worker tool that summarizes persisted run artifacts and metrics.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

from rewardlab.agentic.contracts import ToolResult
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class SummarizeRunArtifactsTool:
    """Build a compact summary over one selected run's evidence."""

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return a summary payload for the selected or latest run."""

        del record, candidates
        if len(runs) == 0:
            return ToolResult(
                status="ok",
                summary="No runs are available to summarize yet.",
                payload={"run_id": None, "summary": "no runs available"},
            )

        selected = _select_run(runs=runs, action_input=action_input)
        metric_keys = sorted(selected.metrics.keys())
        score = _score_from_run(selected.metrics)
        summary = (
            f"Run {selected.run_id} ({selected.status.value}) has score {score:.6f} "
            f"and {len(metric_keys)} metric keys."
        )
        return ToolResult(
            status="ok",
            summary=summary,
            payload={
                "run_id": selected.run_id,
                "candidate_id": selected.candidate_id,
                "status": selected.status.value,
                "score": score,
                "metric_keys": metric_keys,
                "metrics": selected.metrics,
                "artifact_refs": selected.artifact_refs,
                "failure_reason": selected.failure_reason,
            },
        )


def _select_run(*, runs: list[ExperimentRun], action_input: dict[str, object]) -> ExperimentRun:
    """Return the run requested by action input or the latest available run."""

    requested_run_id = action_input.get("run_id")
    if isinstance(requested_run_id, str):
        for run in runs:
            if run.run_id == requested_run_id:
                return run
    return runs[-1]


def _score_from_run(metrics: dict[str, object]) -> float:
    """Extract a canonical score value from a run metric payload."""

    episode_reward = metrics.get("episode_reward")
    if isinstance(episode_reward, (int, float)):
        return float(episode_reward)
    total_reward = metrics.get("total_reward")
    if isinstance(total_reward, (int, float)):
        return float(total_reward)
    return 0.0
