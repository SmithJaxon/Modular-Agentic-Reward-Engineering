"""
Summary: Worker tool for policy-gated human feedback requests.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import datetime, timezone

from rewardlab.agentic.contracts import ToolResult
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class RequestHumanFeedbackTool:
    """Create a traceable human-feedback request for a candidate."""

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return a feedback request envelope or a policy error."""

        del runs
        policy = record.spec.governance.human_feedback
        if not policy.allow:
            return ToolResult(
                status="error",
                summary="Human feedback requests are disabled by governance policy.",
                payload={"requested": False, "reason": "human_feedback_disabled"},
            )

        consumed = record.budget_ledger.consumed_human_feedback_requests
        if policy.max_requests > 0 and consumed >= policy.max_requests:
            return ToolResult(
                status="error",
                summary="Human feedback request budget is exhausted.",
                payload={"requested": False, "reason": "human_feedback_budget_exhausted"},
            )

        target_candidate = _select_candidate(candidates=candidates, action_input=action_input)
        prompt = action_input.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = (
                "Review this candidate for reward shaping quality, exploitability, "
                "and alignment to the stated objective."
            )
        request_id = (
            f"{record.experiment_id}-feedback-request-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        )
        return ToolResult(
            status="ok",
            summary=(
                f"Created human feedback request {request_id} "
                f"for {target_candidate.candidate_id}."
            ),
            payload={
                "requested": True,
                "request_id": request_id,
                "candidate_id": target_candidate.candidate_id,
                "feedback_gate": policy.feedback_gate.value,
                "prompt": prompt,
            },
        )


def _select_candidate(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> RewardCandidate:
    """Choose candidate requested by controller or fallback to best/latest."""

    candidate_id = action_input.get("candidate_id")
    if isinstance(candidate_id, str):
        for candidate in candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
    scored = [candidate for candidate in candidates if candidate.aggregate_score is not None]
    if scored:
        return max(scored, key=lambda candidate: candidate.aggregate_score or float("-inf"))
    return max(candidates, key=lambda candidate: candidate.iteration_index)

