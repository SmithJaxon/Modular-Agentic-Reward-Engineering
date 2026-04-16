"""
Summary: Service facade for agentic run lifecycle, events, and reports.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.decision_loop import DecisionLoop
from rewardlab.agentic.primary_optimizer import PrimaryOptimizer
from rewardlab.agentic.spec_loader import load_run_spec
from rewardlab.agentic.worker_runner import WorkerRunner
from rewardlab.persistence.agentic_repository import AgenticRepository
from rewardlab.tools.broker import ToolBroker, build_default_registry


class AgenticService:
    """
    Coordinate agentic run operations and persisted artifacts.
    """

    def __init__(self, repository: AgenticRepository) -> None:
        """
        Initialize service with run-scoped persistence dependency.
        """
        self._repository = repository

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> AgenticService:
        """
        Build service bound to project-local data directory.
        """
        return cls(repository=AgenticRepository(data_dir))

    def run(self, spec_file: Path, *, run_id: str | None = None) -> dict[str, Any]:
        """
        Start and execute one full agentic run from a spec file.
        """
        spec = load_run_spec(spec_file)
        resolved_run_id = run_id or f"agentrun-{uuid4().hex[:12]}"
        self._repository.create_run(
            run_id=resolved_run_id,
            spec_payload=spec.model_dump(mode="json"),
            spec_path=str(spec_file),
        )
        context = ContextStore()
        budget_engine = BudgetEngine.from_spec(spec)
        broker = ToolBroker(
            registry=build_default_registry(),
            budget_engine=budget_engine,
            enabled_tools=spec.tools.enabled,
        )
        worker_runner = WorkerRunner(tool_broker=broker)
        loop = DecisionLoop(
            primary_optimizer=PrimaryOptimizer(),
            worker_runner=worker_runner,
            repository=self._repository,
        )
        result = loop.execute(
            run_id=resolved_run_id,
            spec=spec,
            context=context,
            budget_engine=budget_engine,
        )
        report_path = self._repository.write_report(resolved_run_id, result.report_payload)
        self._repository.update_run(resolved_run_id, report_path=str(report_path))
        return {
            "run_id": resolved_run_id,
            "status": result.status.value,
            "stop_reason": result.stop_reason,
            "turn_count": result.turn_count,
            "report_path": str(report_path),
        }

    def status(self, run_id: str) -> dict[str, Any]:
        """
        Return persisted status metadata for one run.
        """
        payload = self._repository.get_run(run_id)
        if payload is None:
            raise RuntimeError(f"run not found: {run_id}")
        return payload

    def events(self, run_id: str, *, limit: int | None = None) -> dict[str, Any]:
        """
        Return persisted event rows for one run.
        """
        events = self._repository.list_events(run_id, limit=limit)
        return {"run_id": run_id, "event_count": len(events), "events": events}

    def report(self, run_id: str) -> dict[str, Any]:
        """
        Return persisted report payload for one run.
        """
        report_payload = self._repository.load_report(run_id)
        if report_payload is None:
            raise RuntimeError(f"report not found for run: {run_id}")
        return report_payload
