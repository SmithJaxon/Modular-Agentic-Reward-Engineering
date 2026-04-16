"""
Summary: Minimal context packets for isolated worker tool execution.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rewardlab.schemas.agentic_run import AgentDecision, AgenticRunSpec


@dataclass(slots=True, frozen=True)
class WorkerTaskPacket:
    """
    Carry the minimum fields required for one worker tool execution.
    """

    run_id: str
    turn_index: int
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_rationale: str
    objective_hint: str
    budget_remaining: dict[str, int | float | dict[str, int]]
    requested_at: str


def build_worker_task_packet(
    *,
    run_id: str,
    decision: AgentDecision,
    spec: AgenticRunSpec,
    budget_remaining: dict[str, int | float | dict[str, int]],
) -> WorkerTaskPacket:
    """
    Build a minimal worker packet from a primary-agent tool decision.
    """
    tool_name = decision.tool_name or ""
    return WorkerTaskPacket(
        run_id=run_id,
        turn_index=decision.turn_index,
        tool_name=tool_name,
        tool_arguments=dict(decision.tool_arguments),
        tool_rationale=decision.tool_rationale or "no rationale provided",
        objective_hint=spec.run_name,
        budget_remaining=budget_remaining,
        requested_at=datetime.now(UTC).isoformat(),
    )
