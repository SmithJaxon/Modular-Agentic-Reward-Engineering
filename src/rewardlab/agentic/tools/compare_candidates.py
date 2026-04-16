"""
Summary: Worker tool that compares scored candidates and recommends one.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from typing import Literal, cast

from rewardlab.agentic.contracts import ToolResult
from rewardlab.llm.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class CompareCandidatesTool:
    """Compare evaluated candidates and return a ranking summary."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store optional OpenAI client used by analyzer-assisted comparisons."""

        self.openai_client = openai_client or OpenAIClient()

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return a ranked candidate comparison payload."""

        del runs
        scored = [candidate for candidate in candidates if candidate.aggregate_score is not None]
        if not scored:
            return ToolResult(
                status="ok",
                summary="No scored candidates available to compare.",
                payload={"recommended_candidate_id": None, "ranking": []},
            )

        requested_ids = action_input.get("candidate_ids")
        if isinstance(requested_ids, list):
            requested_id_set = {item for item in requested_ids if isinstance(item, str)}
            if requested_id_set:
                scored = [
                    candidate
                    for candidate in scored
                    if candidate.candidate_id in requested_id_set
                ]

        ranked = sorted(
            scored,
            key=lambda candidate: (
                candidate.aggregate_score
                if candidate.aggregate_score is not None
                else float("-inf"),
                candidate.iteration_index,
            ),
            reverse=True,
        )
        if not ranked:
            return ToolResult(
                status="error",
                summary="Requested candidate_ids did not match any scored candidates.",
                payload={"recommended_candidate_id": None, "ranking": []},
            )

        top = ranked[0]
        top_score = float(top.aggregate_score or 0.0)
        second_score = float(ranked[1].aggregate_score or 0.0) if len(ranked) > 1 else top_score
        ranking_payload: list[dict[str, object]] = [
            {
                "candidate_id": candidate.candidate_id,
                "iteration_index": candidate.iteration_index,
                "aggregate_score": candidate.aggregate_score,
            }
            for candidate in ranked
        ]
        summary = (
            f"Compared {len(ranked)} candidates; "
            f"recommended {top.candidate_id} (score={top_score:.6f})."
        )
        consumed_tokens = 0
        analyzer_reason: str | None = None
        if self.openai_client.has_credentials:
            try:
                analyzer_reason, recommended_id, token_count = _analyzer_compare(
                    openai_client=self.openai_client,
                    record=record,
                    ranking_payload=ranking_payload,
                )
                consumed_tokens = token_count
                if recommended_id is not None:
                    matched = next(
                        (
                            item
                            for item in ranked
                            if item.candidate_id == recommended_id
                        ),
                        None,
                    )
                    if matched is not None:
                        top = matched
                        top_score = float(top.aggregate_score or 0.0)
                if analyzer_reason:
                    summary = f"{summary} Analyzer: {analyzer_reason}"
            except Exception:
                analyzer_reason = None

        return ToolResult(
            status="ok",
            summary=summary,
            payload={
                "recommended_candidate_id": top.candidate_id,
                "top_score": top_score,
                "score_gap_to_second": top_score - second_score,
                "ranking": ranking_payload,
                "analyzer_reason": analyzer_reason,
            },
            consumed_tokens=consumed_tokens,
        )


def _analyzer_compare(
    *,
    openai_client: OpenAIClient,
    record: AgentExperimentRecord,
    ranking_payload: list[dict[str, object]],
) -> tuple[str | None, str | None, int]:
    """Return analyzer summary, recommendation id, and consumed tokens."""

    model_cfg = record.spec.models.analyzer
    prompt = (
        f"Objective: {record.spec.objective}\n"
        f"Environment: {record.spec.environment.id}\n"
        f"Candidate ranking: {json.dumps(ranking_payload)}\n"
        "Choose one candidate and explain briefly.\n"
        "Return JSON with keys recommended_candidate_id and summary."
    )
    response = openai_client.chat_completion(
        ChatCompletionRequest(
            model=model_cfg.model,
            messages=(
                ChatMessage(
                    role="system",
                    content=(
                        "You compare RL reward candidates. Respond with JSON only."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ),
            reasoning_effort=_coerce_reasoning(model_cfg.reasoning_effort),
            max_tokens=min(
                model_cfg.max_completion_tokens,
                record.spec.budgets.api.max_completion_tokens_per_call,
            ),
            response_format={"type": "json_object"},
        )
    )
    payload = json.loads(response.content.strip())
    summary = payload.get("summary")
    recommended_id = payload.get("recommended_candidate_id")
    return (
        summary if isinstance(summary, str) and summary.strip() else None,
        recommended_id if isinstance(recommended_id, str) and recommended_id.strip() else None,
        response.total_tokens or 0,
    )


def _coerce_reasoning(value: str) -> Literal["minimal", "low", "medium", "high"]:
    """Return a valid reasoning effort string for model requests."""

    normalized = value.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"
