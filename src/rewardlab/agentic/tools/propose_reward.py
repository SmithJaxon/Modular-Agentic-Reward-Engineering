"""
Summary: Worker tool that proposes the next reward candidate revision.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from typing import Literal, cast

from rewardlab.agentic.contracts import ToolResult
from rewardlab.llm.openai_client import OpenAIClient
from rewardlab.orchestrator.reward_designer import (
    DeterministicRewardDesigner,
    OpenAIRewardDesigner,
    RewardDesignConfig,
    RewardDesignerMode,
    RewardDesignRequest,
)
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import EnvironmentBackend


class ProposeRewardTool:
    """Generate a revised reward candidate from the latest experiment context."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store OpenAI client used by the model-backed reward designer."""

        self.openai_client = openai_client or OpenAIClient()

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return the next candidate revision result payload."""

        parent_candidate = _select_parent_candidate(
            candidates=candidates,
            action_input=action_input,
        )
        latest_parent_run = _latest_run_for_candidate(
            candidate_id=parent_candidate.candidate_id,
            runs=runs,
        )
        spec = record.spec
        model_cfg = spec.models.reward_designer

        designer = _resolve_designer(
            openai_client=self.openai_client,
            model=model_cfg.model,
            reasoning_effort=model_cfg.reasoning_effort,
            max_tokens=min(
                model_cfg.max_completion_tokens,
                spec.budgets.api.max_completion_tokens_per_call,
            ),
        )
        request = RewardDesignRequest(
            session_id=record.experiment_id,
            objective_text=spec.objective,
            environment_id=spec.environment.id,
            environment_backend=EnvironmentBackend(spec.environment.backend),
            current_candidate=parent_candidate,
            next_iteration_index=parent_candidate.iteration_index + 1,
            latest_reflection=None,
            latest_run=latest_parent_run,
        )
        try:
            result = designer.design_next_candidate(request)
        except Exception:
            result = DeterministicRewardDesigner().design_next_candidate(request)
        candidate = RewardCandidate(
            candidate_id=(
                f"{record.experiment_id}-candidate-"
                f"{parent_candidate.iteration_index + 1:03d}"
            ),
            session_id=record.experiment_id,
            parent_candidate_id=parent_candidate.candidate_id,
            iteration_index=parent_candidate.iteration_index + 1,
            reward_definition=result.reward_definition,
            change_summary=result.change_summary,
            aggregate_score=None,
        )
        return ToolResult(
            status="ok",
            summary=(
                f"Proposed candidate {candidate.candidate_id} from "
                f"{parent_candidate.candidate_id}."
            ),
            payload={"candidate": candidate.model_dump(mode="json")},
        )


def _resolve_designer(
    *,
    openai_client: OpenAIClient,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
) -> OpenAIRewardDesigner | DeterministicRewardDesigner:
    """Resolve a reward designer based on available model credentials."""

    if openai_client.has_credentials:
        return OpenAIRewardDesigner(
            openai_client=openai_client,
            config=RewardDesignConfig(
                mode=RewardDesignerMode.OPENAI,
                model=model,
                reasoning_effort=_coerce_reasoning(reasoning_effort),
                max_tokens=max_tokens,
            ),
        )
    return DeterministicRewardDesigner()


def _coerce_reasoning(
    reasoning_effort: str,
) -> Literal["minimal", "low", "medium", "high"]:
    """Return a valid reasoning effort for RewardDesignConfig."""

    normalized = reasoning_effort.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"


def _select_parent_candidate(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> RewardCandidate:
    """Select the parent candidate for reward revision."""

    requested_parent = action_input.get("parent_candidate_id")
    if isinstance(requested_parent, str):
        for candidate in candidates:
            if candidate.candidate_id == requested_parent:
                return candidate

    evaluated = [candidate for candidate in candidates if candidate.aggregate_score is not None]
    if evaluated:
        return max(evaluated, key=lambda candidate: candidate.aggregate_score or float("-inf"))
    return max(candidates, key=lambda candidate: candidate.iteration_index)


def _latest_run_for_candidate(
    *,
    candidate_id: str,
    runs: list[ExperimentRun],
) -> ExperimentRun | None:
    """Return the latest run recorded for a candidate."""

    matching = [run for run in runs if run.candidate_id == candidate_id]
    if not matching:
        return None
    return matching[-1]
