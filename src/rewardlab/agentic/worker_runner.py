"""
Summary: Worker runner that executes brokered tool calls from isolated task packets.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rewardlab.agentic.worker_context import WorkerTaskPacket
from rewardlab.schemas.tool_contracts import ToolResult
from rewardlab.tools.broker import ToolBroker


@dataclass(slots=True, frozen=True)
class WorkerExecutionRecord:
    """
    Capture one worker execution envelope around a tool result.
    """

    packet: WorkerTaskPacket
    started_at: str
    finished_at: str
    result: ToolResult


class WorkerRunner:
    """
    Execute worker task packets through the shared tool broker.
    """

    def __init__(self, *, tool_broker: ToolBroker) -> None:
        """
        Initialize worker runner with the broker dependency.
        """
        self._tool_broker = tool_broker

    def execute(self, packet: WorkerTaskPacket) -> WorkerExecutionRecord:
        """
        Execute one isolated worker task packet and return execution record.
        """
        started_at = datetime.now(UTC).isoformat()
        result = self._tool_broker.execute(
            turn_index=packet.turn_index,
            tool_name=packet.tool_name,
            arguments=packet.tool_arguments,
            rationale=packet.tool_rationale,
        )
        finished_at = datetime.now(UTC).isoformat()
        return WorkerExecutionRecord(
            packet=packet,
            started_at=started_at,
            finished_at=finished_at,
            result=result,
        )
