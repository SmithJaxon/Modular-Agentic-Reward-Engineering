"""
Summary: Decision-turn runtime loop for agentic optimization sessions.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.llm_planner import PlannerAttemptFeedback
from rewardlab.agentic.primary_optimizer import PrimaryOptimizer
from rewardlab.agentic.worker_context import build_worker_task_packet
from rewardlab.agentic.worker_runner import WorkerRunner
from rewardlab.persistence.agentic_repository import AgenticRepository
from rewardlab.schemas.agentic_run import (
    AgentDecisionAction,
    AgenticRunSpec,
    AgenticRunStatus,
    StopDecisionTag,
)


@dataclass(slots=True, frozen=True)
class DecisionLoopResult:
    """
    Capture terminal decision-loop status and report payload.
    """

    run_id: str
    status: AgenticRunStatus
    stop_reason: str
    turn_count: int
    report_payload: dict[str, Any]


class DecisionLoop:
    """
    Execute a primary-agent decision loop without fixed tool-call ordering.
    """

    def __init__(
        self,
        *,
        primary_optimizer: PrimaryOptimizer,
        worker_runner: WorkerRunner,
        repository: AgenticRepository,
    ) -> None:
        """
        Initialize runtime dependencies for one loop execution.
        """
        self._primary_optimizer = primary_optimizer
        self._worker_runner = worker_runner
        self._repository = repository

    def execute(
        self,
        *,
        run_id: str,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_engine: BudgetEngine,
    ) -> DecisionLoopResult:
        """
        Execute decision turns until stop decision or turn cap is reached.
        """
        final_stop_reason = StopDecisionTag.TURN_CAP.value
        turns_executed = 0
        planner_feedback_rows: list[dict[str, Any]] = []
        for turn_index in range(spec.decision.max_turns):
            decision = self._primary_optimizer.decide(
                run_id=run_id,
                turn_index=turn_index,
                spec=spec,
                context=context,
                budget_state=budget_engine.state,
            )
            for feedback in self._primary_optimizer.drain_planner_feedback():
                feedback_payload = _planner_feedback_payload(
                    turn_index=turn_index,
                    feedback=feedback,
                )
                planner_feedback_rows.append(feedback_payload)
                self._repository.append_event(
                    run_id=run_id,
                    event_type="planner.validation_failed",
                    payload=feedback_payload,
                )
            context.record_decision(decision)
            self._repository.append_event(
                run_id=run_id,
                event_type="agent.decision",
                payload=decision.model_dump(mode="json"),
            )
            turns_executed = turn_index + 1

            if decision.action is AgentDecisionAction.STOP:
                final_stop_reason = (
                    decision.stop_tag.value if decision.stop_tag else StopDecisionTag.MANUAL.value
                )
                break

            if decision.action is AgentDecisionAction.REQUEST_TOOL and decision.tool_name:
                worker_packet = build_worker_task_packet(
                    run_id=run_id,
                    decision=decision,
                    spec=spec,
                    budget_remaining=budget_engine.remaining(),
                )
                self._repository.append_event(
                    run_id=run_id,
                    event_type="worker.task_started",
                    payload={
                        "turn_index": turn_index,
                        "packet": asdict(worker_packet),
                    },
                )
                worker_record = self._worker_runner.execute(worker_packet)
                result = worker_record.result
                context.record_tool_result(result)
                self._repository.append_event(
                    run_id=run_id,
                    event_type="tool.result",
                    payload=result.model_dump(mode="json"),
                )
                self._repository.append_event(
                    run_id=run_id,
                    event_type="worker.task_completed",
                    payload={
                        "turn_index": turn_index,
                        "started_at": worker_record.started_at,
                        "finished_at": worker_record.finished_at,
                        "status": result.status.value,
                        "tool_name": result.tool_name,
                    },
                )
                self._repository.update_run(
                    run_id,
                    turn_count=turns_executed,
                    budget_remaining=budget_engine.remaining(),
                )
                continue

            self._repository.update_run(
                run_id,
                turn_count=turns_executed,
                budget_remaining=budget_engine.remaining(),
            )

        report = {
            "run_id": run_id,
            "status": AgenticRunStatus.COMPLETED.value,
            "stop_reason": final_stop_reason,
            "turn_count": turns_executed,
            "best_score": context.best_score,
            "decision_count": len(context.decisions),
            "tool_result_count": len(context.tool_results),
            "remaining_budget": budget_engine.remaining(),
            "planner_feedback": planner_feedback_rows,
            "planner_feedback_summary": _summarize_planner_feedback(planner_feedback_rows),
            "decisions": [item.model_dump(mode="json") for item in context.decisions],
            "tool_results": [item.model_dump(mode="json") for item in context.tool_results],
        }
        self._repository.update_run(
            run_id,
            status=AgenticRunStatus.COMPLETED.value,
            stop_reason=final_stop_reason,
            turn_count=turns_executed,
            budget_remaining=budget_engine.remaining(),
        )
        return DecisionLoopResult(
            run_id=run_id,
            status=AgenticRunStatus.COMPLETED,
            stop_reason=final_stop_reason,
            turn_count=turns_executed,
            report_payload=report,
        )


def _planner_feedback_payload(
    *,
    turn_index: int,
    feedback: PlannerAttemptFeedback,
) -> dict[str, Any]:
    """
    Convert one planner feedback row into a JSON-serializable payload.
    """
    return {
        "turn_index": turn_index,
        "attempt_index": feedback.attempt_index,
        "max_attempts": feedback.max_attempts,
        "failure_type": feedback.failure_type,
        "reason": feedback.reason,
        "output_excerpt": feedback.output_excerpt,
    }


def _summarize_planner_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build compact planner failure summary counts for run reports.
    """
    by_failure_type: dict[str, int] = {}
    for row in rows:
        failure_type_raw = row.get("failure_type")
        if not isinstance(failure_type_raw, str) or not failure_type_raw:
            continue
        by_failure_type[failure_type_raw] = by_failure_type.get(failure_type_raw, 0) + 1
    return {
        "failure_count": len(rows),
        "by_failure_type": by_failure_type,
    }
