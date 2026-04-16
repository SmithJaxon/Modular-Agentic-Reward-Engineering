"""
Summary: Primary decision controller for autonomous tool-calling experiments.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from rewardlab.agentic.contracts import ControllerAction
from rewardlab.llm.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from rewardlab.schemas.agent_experiment import (
    ActionType,
    AgentExperimentRecord,
    AgentExperimentSpec,
)
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate

ACTION_TOOL_NAME: dict[ActionType, str] = {
    ActionType.RUN_EXPERIMENT: "run_experiment",
    ActionType.RUN_ROBUSTNESS_PROBES: "run_robustness_probes",
    ActionType.PROPOSE_REWARD: "propose_reward_revision",
    ActionType.SUMMARIZE_RUN_ARTIFACTS: "summarize_run_artifacts",
    ActionType.VALIDATE_REWARD_PROGRAM: "validate_reward_program",
    ActionType.ESTIMATE_COST_AND_RISK: "estimate_cost_and_risk",
    ActionType.COMPARE_CANDIDATES: "compare_candidates",
    ActionType.REQUEST_HUMAN_FEEDBACK: "request_human_feedback",
    ActionType.STOP: "stop_or_continue_recommendation",
}


@dataclass(frozen=True)
class ControllerContext:
    """Compact, rich context bundle passed to the controller for one decision."""

    record: AgentExperimentRecord
    candidates: list[RewardCandidate]
    runs: list[ExperimentRun]
    recent_decisions: list[dict[str, Any]]
    failed_actions: int
    no_improve_streak: int


class ControllerAgent:
    """Primary strategy controller that chooses the next experiment action."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store the OpenAI client used for model-backed decisions."""

        self.openai_client = openai_client or OpenAIClient()

    def choose_action(self, context: ControllerContext) -> tuple[ControllerAction, int]:
        """Return the next action and observed token usage for this decision."""

        if not self.openai_client.has_credentials:
            return self._heuristic_action(context), 0

        model_config = context.record.spec.models.controller
        prompt = _build_controller_prompt(context)
        try:
            response = self.openai_client.chat_completion(
                ChatCompletionRequest(
                    model=model_config.model,
                    messages=(
                        ChatMessage(
                            role="system",
                            content=(
                                "You are the primary RL experiment controller. "
                                "Choose exactly one next action. Respond with JSON only."
                            ),
                        ),
                        ChatMessage(role="user", content=prompt),
                    ),
                    reasoning_effort=_coerce_reasoning(model_config.reasoning_effort),
                    max_tokens=min(
                        model_config.max_completion_tokens,
                        context.record.spec.budgets.api.max_completion_tokens_per_call,
                    ),
                    response_format={"type": "json_object"},
                )
            )
        except Exception:
            return self._heuristic_action(context), 0
        try:
            payload = json.loads(response.content.strip())
            action = ControllerAction.model_validate(payload)
        except Exception:
            action = self._heuristic_action(context)
        if not _is_action_permitted(action, context.record):
            action = self._heuristic_action(context)
        total_tokens = response.total_tokens or 0
        return action, total_tokens

    def _heuristic_action(self, context: ControllerContext) -> ControllerAction:
        """Return a deterministic fallback action when model output is unavailable."""

        candidates = sorted(context.candidates, key=lambda candidate: candidate.iteration_index)
        latest = candidates[-1]
        evaluated_ids = {run.candidate_id for run in context.runs}
        spec = context.record.spec
        stopping = spec.governance.stopping
        recent_action_types = [
            item.get("action_type")
            for item in context.recent_decisions[-3:]
            if isinstance(item, dict)
        ]
        token_utilization = (
            context.record.budget_ledger.consumed_total_tokens / spec.budgets.api.max_total_tokens
        )
        experiment_utilization = (
            context.record.budget_ledger.consumed_experiments / spec.budgets.compute.max_experiments
        )
        scored_candidates = [
            candidate for candidate in candidates if candidate.aggregate_score is not None
        ]
        latest_completed_performance_runs = [
            run
            for run in context.runs
            if run.candidate_id == latest.candidate_id
            and run.run_type.value == "performance"
            and run.status.value == "completed"
        ]
        latest_has_robustness_probe = any(
            run.candidate_id == latest.candidate_id and run.run_type.value == "robustness"
            for run in context.runs
        )

        if context.failed_actions >= stopping.max_failed_actions:
            return ControllerAction(
                action_type=ActionType.STOP,
                rationale="Stopping after repeated failed actions.",
                expected_value=0.0,
                expected_cost=0.0,
            )

        if (
            (token_utilization >= 0.8 or experiment_utilization >= 0.8)
            and "estimate_cost_and_risk" not in recent_action_types
            and _is_action_type_allowed(
                ActionType.ESTIMATE_COST_AND_RISK,
                spec.tool_policy.allowed_tools,
            )
        ):
            return ControllerAction(
                action_type=ActionType.ESTIMATE_COST_AND_RISK,
                rationale="Budget utilization is elevated; estimating risk before new spend.",
                expected_value=0.25,
                expected_cost=0.02,
            )

        if latest.candidate_id not in evaluated_ids:
            return ControllerAction(
                action_type=ActionType.RUN_EXPERIMENT,
                rationale="Latest candidate has not been evaluated yet.",
                expected_value=0.7,
                expected_cost=0.5,
                action_input={"candidate_id": latest.candidate_id},
            )

        if latest.iteration_index >= stopping.max_iterations:
            return ControllerAction(
                action_type=ActionType.STOP,
                rationale="Iteration cap reached.",
                expected_value=0.0,
                expected_cost=0.0,
            )

        feedback_policy = spec.governance.human_feedback
        if (
            feedback_policy.allow
            and context.record.budget_ledger.consumed_human_feedback_requests
            < feedback_policy.max_requests
            and context.no_improve_streak > 0
            and "request_human_feedback" not in recent_action_types
            and _is_action_type_allowed(
                ActionType.REQUEST_HUMAN_FEEDBACK,
                spec.tool_policy.allowed_tools,
            )
            and scored_candidates
        ):
            return ControllerAction(
                action_type=ActionType.REQUEST_HUMAN_FEEDBACK,
                rationale="No-improve streak suggests getting external signal on best candidate.",
                expected_value=0.2,
                expected_cost=0.05,
                action_input={
                    "candidate_id": max(
                        scored_candidates,
                        key=lambda candidate: candidate.aggregate_score or float("-inf"),
                    ).candidate_id
                },
            )

        if (
            len(scored_candidates) >= 2
            and context.no_improve_streak > 0
            and "compare_candidates" not in recent_action_types
            and _is_action_type_allowed(
                ActionType.COMPARE_CANDIDATES,
                spec.tool_policy.allowed_tools,
            )
        ):
            return ControllerAction(
                action_type=ActionType.COMPARE_CANDIDATES,
                rationale=(
                    "Recent performance stalled; compare top candidates "
                    "before next mutation."
                ),
                expected_value=0.3,
                expected_cost=0.03,
            )

        if (
            context.no_improve_streak > 0
            and len(latest_completed_performance_runs) > 0
            and not latest_has_robustness_probe
            and "run_robustness_probes" not in recent_action_types
            and _is_action_type_allowed(
                ActionType.RUN_ROBUSTNESS_PROBES,
                spec.tool_policy.allowed_tools,
            )
        ):
            primary_run = latest_completed_performance_runs[-1]
            return ControllerAction(
                action_type=ActionType.RUN_ROBUSTNESS_PROBES,
                rationale=(
                    "Recent stall warrants robustness probing to check for reward hacking "
                    "or brittle overfitting."
                ),
                expected_value=0.35,
                expected_cost=0.25,
                action_input={
                    "candidate_id": latest.candidate_id,
                    "primary_run_id": primary_run.run_id,
                },
            )

        if context.no_improve_streak >= stopping.max_no_improve_streak:
            return ControllerAction(
                action_type=ActionType.STOP,
                rationale="No-improvement streak reached configured threshold.",
                expected_value=0.0,
                expected_cost=0.0,
            )

        return ControllerAction(
            action_type=ActionType.PROPOSE_REWARD,
            rationale="Need a revised reward candidate for the next evaluation.",
            expected_value=0.6,
            expected_cost=0.2,
            action_input={"parent_candidate_id": latest.candidate_id},
        )


def _build_controller_prompt(context: ControllerContext) -> str:
    """Build a controller decision prompt from compact experiment state."""

    spec = context.record.spec
    candidates = sorted(context.candidates, key=lambda item: item.iteration_index)
    latest_candidate = candidates[-1]
    best_candidate = max(
        candidates,
        key=lambda item: (
            item.aggregate_score if item.aggregate_score is not None else float("-inf")
        ),
    )
    scores = [
        {
            "candidate_id": candidate.candidate_id,
            "iteration_index": candidate.iteration_index,
            "aggregate_score": candidate.aggregate_score,
        }
        for candidate in candidates[-6:]
    ]
    recent_runs = [
        {
            "run_id": run.run_id,
            "candidate_id": run.candidate_id,
            "status": run.status.value,
            "metrics": run.metrics,
        }
        for run in context.runs[-3:]
    ]

    return (
        f"Experiment id: {context.record.experiment_id}\n"
        f"Objective: {spec.objective}\n"
        f"Environment: {spec.environment.backend.value}:{spec.environment.id}\n"
        f"Latest candidate: {latest_candidate.candidate_id}\n"
        f"Best candidate: {best_candidate.candidate_id}\n"
        f"No-improve streak: {context.no_improve_streak}\n"
        f"Failed actions: {context.failed_actions}\n"
        f"Budget ledger: tokens={context.record.budget_ledger.consumed_total_tokens}, "
        f"usd={context.record.budget_ledger.consumed_total_usd}, "
        f"experiments={context.record.budget_ledger.consumed_experiments}, "
        f"timesteps={context.record.budget_ledger.consumed_train_timesteps}\n"
        f"Budget limits: tokens={spec.budgets.api.max_total_tokens}, "
        f"usd={spec.budgets.api.max_total_usd}, "
        f"experiments={spec.budgets.compute.max_experiments}, "
        f"timesteps={spec.budgets.compute.max_total_train_timesteps}\n"
        f"Recent candidate scores: {json.dumps(scores)}\n"
        f"Recent runs: {json.dumps(recent_runs)}\n"
        f"Allowed actions: {', '.join(_allowed_action_names(spec=spec))}\n"
        "Return JSON with keys action_type, rationale, expected_value, "
        "expected_cost, action_input.\n"
        "Choose stop when expected gain is low versus budget risk or plateau persists.\n"
    )


def _coerce_reasoning(value: str) -> Literal["minimal", "low", "medium", "high"]:
    """Return a valid reasoning effort string."""

    normalized = value.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"


def _allowed_action_names(*, spec: AgentExperimentSpec) -> list[str]:
    """Return sorted action names permitted by tool policy and governance."""

    allowed: list[str] = []
    for action_type, tool_name in ACTION_TOOL_NAME.items():
        if tool_name not in spec.tool_policy.allowed_tools:
            continue
        if (
            action_type == ActionType.REQUEST_HUMAN_FEEDBACK
            and not spec.governance.human_feedback.allow
        ):
            continue
        allowed.append(action_type.value)
    return sorted(allowed)


def _is_action_type_allowed(action_type: ActionType, allowed_tools: list[str]) -> bool:
    """Return whether action's mapped tool is allowed."""

    return ACTION_TOOL_NAME[action_type] in allowed_tools


def _is_action_permitted(action: ControllerAction, record: AgentExperimentRecord) -> bool:
    """Return whether the chosen action respects tool and governance policy."""

    if not _is_action_type_allowed(action.action_type, record.spec.tool_policy.allowed_tools):
        return False
    return not (
        action.action_type == ActionType.REQUEST_HUMAN_FEEDBACK
        and not record.spec.governance.human_feedback.allow
    )
