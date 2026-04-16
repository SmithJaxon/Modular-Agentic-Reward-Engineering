"""
Summary: Worker tool that estimates remaining budget and stop-risk signals.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal, cast

from rewardlab.agentic.contracts import ToolResult
from rewardlab.llm.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class EstimateCostAndRiskTool:
    """Estimate budget headroom and return a compact risk assessment."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store optional OpenAI client used for analyzer-assisted risk scoring."""

        self.openai_client = openai_client or OpenAIClient()

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return remaining budget estimates and aggregate risk level."""

        del candidates, runs, action_input
        spec = record.spec
        ledger = record.budget_ledger
        started_at = record.started_at or record.created_at
        elapsed_minutes = max((datetime.now(UTC) - started_at).total_seconds() / 60.0, 0.0)

        limits = {
            "tokens": float(spec.budgets.api.max_total_tokens),
            "usd": float(spec.budgets.api.max_total_usd),
            "experiments": float(spec.budgets.compute.max_experiments),
            "timesteps": float(spec.budgets.compute.max_total_train_timesteps),
            "minutes": float(spec.budgets.time.max_wall_clock_minutes),
        }
        used = {
            "tokens": float(ledger.consumed_total_tokens),
            "usd": float(ledger.consumed_total_usd),
            "experiments": float(ledger.consumed_experiments),
            "timesteps": float(ledger.consumed_train_timesteps),
            "minutes": elapsed_minutes,
        }
        remaining = {key: max(limits[key] - used[key], 0.0) for key in limits}
        utilization = {
            key: (used[key] / limits[key]) if limits[key] > 0 else 0.0
            for key in limits
        }
        risk_level = _aggregate_risk_level(utilization)
        recommend_stop = risk_level == "high"
        analyzer_reason: str | None = None
        consumed_tokens = 0
        if self.openai_client.has_credentials:
            try:
                analyzed = _analyzer_assess_risk(
                    openai_client=self.openai_client,
                    record=record,
                    utilization=utilization,
                    remaining=remaining,
                )
                risk_level = analyzed.risk_level
                recommend_stop = analyzed.recommend_stop
                analyzer_reason = analyzed.summary
                consumed_tokens = analyzed.consumed_tokens
            except Exception:
                analyzer_reason = None

        return ToolResult(
            status="ok",
            summary=(
                f"Estimated budget risk={risk_level}; "
                f"remaining experiments={int(remaining['experiments'])}, "
                f"remaining tokens={int(remaining['tokens'])}."
            ),
            payload={
                "risk_level": risk_level,
                "recommend_stop": recommend_stop,
                "budget_used": used,
                "budget_limits": limits,
                "budget_remaining": remaining,
                "utilization": utilization,
                "analyzer_reason": analyzer_reason,
            },
            consumed_tokens=consumed_tokens,
        )


def _aggregate_risk_level(utilization: dict[str, float]) -> str:
    """Return high/medium/low based on worst-case budget utilization."""

    highest = max(utilization.values(), default=0.0)
    if highest >= 0.9:
        return "high"
    if highest >= 0.75:
        return "medium"
    return "low"


class AnalyzerRiskAssessment:
    """Parsed analyzer output for cost/risk review."""

    def __init__(
        self,
        *,
        risk_level: str,
        recommend_stop: bool,
        summary: str | None,
        consumed_tokens: int,
    ) -> None:
        """Store normalized analyzer response fields."""

        self.risk_level = risk_level
        self.recommend_stop = recommend_stop
        self.summary = summary
        self.consumed_tokens = consumed_tokens


def _analyzer_assess_risk(
    *,
    openai_client: OpenAIClient,
    record: AgentExperimentRecord,
    utilization: dict[str, float],
    remaining: dict[str, float],
) -> AnalyzerRiskAssessment:
    """Return analyzer-adjusted risk estimate and stop recommendation."""

    model_cfg = record.spec.models.analyzer
    prompt = (
        f"Objective: {record.spec.objective}\n"
        f"Environment: {record.spec.environment.id}\n"
        f"Budget utilization: {json.dumps(utilization)}\n"
        f"Budget remaining: {json.dumps(remaining)}\n"
        "Return JSON with keys risk_level (low|medium|high), "
        "recommend_stop (bool), and summary."
    )
    response = openai_client.chat_completion(
        ChatCompletionRequest(
            model=model_cfg.model,
            messages=(
                ChatMessage(
                    role="system",
                    content=(
                        "You assess cost/risk for RL experiments. Respond with JSON only."
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
    risk_level = payload.get("risk_level")
    if not isinstance(risk_level, str):
        risk_level = "medium"
    risk_level = risk_level.strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"
    recommend_stop = payload.get("recommend_stop")
    if not isinstance(recommend_stop, bool):
        recommend_stop = risk_level == "high"
    summary = payload.get("summary")
    return AnalyzerRiskAssessment(
        risk_level=risk_level,
        recommend_stop=recommend_stop,
        summary=summary if isinstance(summary, str) and summary.strip() else None,
        consumed_tokens=response.total_tokens or 0,
    )


def _coerce_reasoning(value: str) -> Literal["minimal", "low", "medium", "high"]:
    """Return a valid reasoning effort string for model requests."""

    normalized = value.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"
