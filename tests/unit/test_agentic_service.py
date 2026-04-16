"""
Summary: Unit tests for autonomous experiment service control loop behavior.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.controller import ControllerContext
from rewardlab.agentic.service import AgentExperimentService
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.orchestrator.session_service import ServicePaths
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.agent_experiment import ActionType


@dataclass
class FakeController:
    """Controller fake that always decides to stop immediately."""

    def choose_action(self, context: ControllerContext) -> tuple[ControllerAction, int]:
        """Return a deterministic stop action with synthetic token usage."""

        del context
        return (
            ControllerAction(
                action_type=ActionType.STOP,
                rationale="done",
                expected_value=0.0,
                expected_cost=0.0,
                action_input={},
            ),
            12,
        )


@dataclass
class FakeBroker:
    """Tool broker fake that honors stop with a successful result payload."""

    def execute_action(self, **kwargs: object) -> ToolResult:
        """Return a deterministic successful stop result."""

        del kwargs
        return ToolResult(
            status="ok",
            summary="controller requested stop",
            payload={"stop": True},
        )


@dataclass
class SequenceController:
    """Controller fake that emits a fixed action sequence."""

    actions: list[ControllerAction]
    cursor: int = 0

    def choose_action(self, context: ControllerContext) -> tuple[ControllerAction, int]:
        """Return the next configured action and synthetic token usage."""

        del context
        index = min(self.cursor, len(self.actions) - 1)
        action = self.actions[index]
        self.cursor += 1
        return action, 5


@dataclass
class FeedbackThenStopBroker:
    """Broker fake that supports feedback requests and stop actions."""

    def execute_action(self, **kwargs: object) -> ToolResult:
        """Return deterministic tool payloads keyed by action type."""

        action = kwargs["action"]
        candidates = kwargs["candidates"]
        action_type = action.action_type.value
        if action_type == "request_human_feedback":
            candidate_id = candidates[0].candidate_id
            return ToolResult(
                status="ok",
                summary="feedback requested",
                payload={
                    "requested": True,
                    "request_id": "feedback-001",
                    "candidate_id": candidate_id,
                    "feedback_gate": "one_required",
                },
            )
        return ToolResult(status="ok", summary="stop", payload={"stop": True})


def _service_for_runtime(runtime_root: Path) -> tuple[ServicePaths, SessionRepository]:
    """Create reusable runtime paths and repository for tests."""

    paths = ServicePaths(
        data_dir=runtime_root,
        database_path=runtime_root / "metadata.sqlite3",
        event_log_dir=runtime_root / "events",
        checkpoint_dir=runtime_root / "checkpoints",
        report_dir=runtime_root / "reports",
    )
    repository = SessionRepository(
        RepositoryPaths(
            database_path=paths.database_path,
            event_log_path=paths.event_log_dir / "events.jsonl",
        )
    )
    return paths, repository


def test_run_experiment_stops_and_writes_trace() -> None:
    """Service should persist baseline state, decision trace, and final report."""

    runtime_root = Path(".agentic-test-runtime")
    paths, repository = _service_for_runtime(runtime_root)
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=FakeController(),
        tool_broker=FakeBroker(),  # type: ignore[arg-type]
    )
    service.initialize()

    result = service.run_experiment(
        spec_file=Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml"),
        experiment_id=f"experiment-test-{uuid4().hex[:8]}",
    )

    assert result.status.value == "completed"
    assert result.stop_reason == "controller_stop"
    assert result.report_path is not None

    status = service.get_status(experiment_id=result.experiment_id)
    assert status.consumed_total_tokens == 12
    assert status.consumed_human_feedback_requests == 0
    trace = service.trace_payload(experiment_id=result.experiment_id)
    assert len(trace["decisions"]) == 1


def test_feedback_request_consumes_feedback_budget() -> None:
    """Service should pause and account for successful human-feedback requests."""

    runtime_root = Path(".agentic-test-runtime-feedback")
    paths, repository = _service_for_runtime(runtime_root)
    spec_payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml")
    ).model_dump(mode="python")
    spec_payload["tool_policy"]["allowed_tools"].append("request_human_feedback")
    spec_payload["governance"]["human_feedback"] = {
        "allow": True,
        "feedback_gate": "one_required",
        "max_requests": 1,
    }
    spec_file = runtime_root / "agent_feedback_enabled.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    controller = SequenceController(
        actions=[
            ControllerAction(
                action_type=ActionType.REQUEST_HUMAN_FEEDBACK,
                rationale="ask for review",
                expected_value=0.1,
                expected_cost=0.0,
                action_input={},
            ),
            ControllerAction(
                action_type=ActionType.STOP,
                rationale="done",
                expected_value=0.0,
                expected_cost=0.0,
                action_input={},
            ),
        ]
    )
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=controller,  # type: ignore[arg-type]
        tool_broker=FeedbackThenStopBroker(),  # type: ignore[arg-type]
    )
    service.initialize()
    result = service.run_experiment(
        spec_file=spec_file,
        experiment_id=f"experiment-feedback-{uuid4().hex[:8]}",
    )

    assert result.status.value == "paused"
    assert result.stop_reason == "awaiting_human_feedback"
    status = service.get_status(experiment_id=result.experiment_id)
    assert status.consumed_human_feedback_requests == 1


def test_submit_feedback_then_resume_completes_loop() -> None:
    """Submitting feedback should unblock pause and allow resume to continue."""

    runtime_root = Path(".agentic-test-runtime-feedback-resume")
    paths, repository = _service_for_runtime(runtime_root)
    spec_payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml")
    ).model_dump(mode="python")
    spec_payload["tool_policy"]["allowed_tools"].append("request_human_feedback")
    spec_payload["governance"]["human_feedback"] = {
        "allow": True,
        "feedback_gate": "one_required",
        "max_requests": 1,
    }
    spec_file = runtime_root / "agent_feedback_resume.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    controller = SequenceController(
        actions=[
            ControllerAction(
                action_type=ActionType.REQUEST_HUMAN_FEEDBACK,
                rationale="ask for review",
                expected_value=0.1,
                expected_cost=0.0,
                action_input={},
            ),
            ControllerAction(
                action_type=ActionType.STOP,
                rationale="done",
                expected_value=0.0,
                expected_cost=0.0,
                action_input={},
            ),
        ]
    )
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=controller,  # type: ignore[arg-type]
        tool_broker=FeedbackThenStopBroker(),  # type: ignore[arg-type]
    )
    service.initialize()
    result = service.run_experiment(
        spec_file=spec_file,
        experiment_id=f"experiment-feedback-resume-{uuid4().hex[:8]}",
    )
    assert result.status.value == "paused"

    trace = service.trace_payload(experiment_id=result.experiment_id)
    request_id = str(trace["feedback_requests"][0]["request_id"])
    submitted = service.submit_human_feedback(
        experiment_id=result.experiment_id,
        candidate_id=f"{result.experiment_id}-candidate-000",
        comment="Looks aligned; continue.",
        request_id=request_id,
    )
    assert submitted.status.value == "paused"

    resumed = service.resume_experiment(experiment_id=result.experiment_id)
    assert resumed.status.value == "completed"
    assert resumed.stop_reason == "controller_stop"


def test_run_benchmark_writes_aggregate_report() -> None:
    """Benchmark runner should execute multiple seeds and emit report artifact."""

    runtime_root = Path(".agentic-test-runtime-benchmark")
    paths, repository = _service_for_runtime(runtime_root)
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=FakeController(),
        tool_broker=FakeBroker(),  # type: ignore[arg-type]
    )
    service.initialize()

    result = service.run_benchmark(
        spec_file=Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml"),
        seeds=[3, 5],
        benchmark_id=f"benchmark-test-{uuid4().hex[:8]}",
    )

    assert result.run_count == 2
    assert result.completed_count == 2
    report_path = Path(result.report_path)
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == result.benchmark_id
    assert payload["seeds"] == [3, 5]
