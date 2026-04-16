"""
Summary: Integration test for mixed-action Humanoid autonomous experiment control.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rewardlab.agentic.contracts import ControllerAction
from rewardlab.agentic.controller import ControllerContext
from rewardlab.agentic.service import AgentExperimentService
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.orchestrator.session_service import ServicePaths
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.agent_experiment import ActionType
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType


@dataclass
class FakeExecutionResult:
    """Minimal execution result shape consumed by `RunExperimentTool`."""

    run: ExperimentRun


@dataclass
class FakeExecutionService:
    """Execution service double returning deterministic completed runs."""

    def execute_candidate(self, **kwargs: object) -> FakeExecutionResult:
        """Return one completed run for the requested candidate."""

        candidate = kwargs["candidate"]
        request = kwargs["request"]
        run = ExperimentRun(
            run_id=request.run_id,
            candidate_id=candidate.candidate_id,
            backend=request.backend,
            environment_id=request.environment_id,
            run_type=RunType.PERFORMANCE,
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            status=RunStatus.COMPLETED,
            metrics={
                "episode_reward": 0.8125,
                "train_timesteps": 50_000,
                "evaluation_protocol": "humanoid_ppo_max_checkpoint_mean_x_velocity",
            },
            artifact_refs=["manifest.json", "metrics.json"],
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        return FakeExecutionResult(run=run)


@dataclass
class SequenceController:
    """Controller double that emits a fixed sequence of actions."""

    actions: list[ControllerAction]
    cursor: int = 0

    def choose_action(self, context: ControllerContext) -> tuple[ControllerAction, int]:
        """Return next action from the sequence and synthetic token usage."""

        del context
        index = min(self.cursor, len(self.actions) - 1)
        action = self.actions[index]
        self.cursor += 1
        return action, 7


def _service_for_runtime(runtime_root: Path) -> tuple[ServicePaths, SessionRepository]:
    """Create reusable runtime paths and repository for integration tests."""

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


def test_agentic_humanoid_mixed_actions_complete_with_full_trace() -> None:
    """Humanoid autonomous loop should persist a mixed action trace and stop cleanly."""

    runtime_root = Path(".agentic-test-runtime-humanoid-mixed")
    runtime_root.mkdir(parents=True, exist_ok=True)

    spec_payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_humanoid_balanced.yaml")
    ).model_dump(mode="python")
    spec_payload["tool_policy"]["mcp_execution_mode"] = "off"
    spec_payload["budgets"]["compute"]["max_experiments"] = 3
    spec_file = runtime_root / "humanoid_mixed.json"
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    experiment_id = f"humanoid-mixed-{uuid4().hex[:8]}"
    candidate_000 = f"{experiment_id}-candidate-000"
    candidate_001 = f"{experiment_id}-candidate-001"

    controller = SequenceController(
        actions=[
            ControllerAction(
                action_type=ActionType.PROPOSE_REWARD,
                rationale="Propose one revised candidate.",
                action_input={"parent_candidate_id": candidate_000},
            ),
            ControllerAction(
                action_type=ActionType.VALIDATE_REWARD_PROGRAM,
                rationale="Validate proposed candidate before running.",
                action_input={"candidate_id": candidate_001},
            ),
            ControllerAction(
                action_type=ActionType.RUN_EXPERIMENT,
                rationale="Run validated candidate.",
                action_input={"candidate_id": candidate_001},
            ),
            ControllerAction(
                action_type=ActionType.SUMMARIZE_RUN_ARTIFACTS,
                rationale="Summarize latest run evidence.",
            ),
            ControllerAction(
                action_type=ActionType.ESTIMATE_COST_AND_RISK,
                rationale="Estimate budget and risk trajectory.",
            ),
            ControllerAction(
                action_type=ActionType.COMPARE_CANDIDATES,
                rationale="Compare baseline and revised candidate.",
            ),
            ControllerAction(
                action_type=ActionType.STOP,
                rationale="Stop after mixed-action evaluation pass.",
            ),
        ]
    )
    paths, repository = _service_for_runtime(runtime_root / "runtime")
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        execution_service=FakeExecutionService(),  # type: ignore[arg-type]
        controller=controller,  # type: ignore[arg-type]
    )
    service.initialize()

    result = service.run_experiment(spec_file=spec_file, experiment_id=experiment_id)

    assert result.status.value == "completed"
    assert result.stop_reason == "controller_stop"
    trace = service.trace_payload(experiment_id=experiment_id)
    action_types = [item["action_type"] for item in trace["decisions"]]
    assert action_types == [
        "propose_reward",
        "validate_reward_program",
        "run_experiment",
        "summarize_run_artifacts",
        "estimate_cost_and_risk",
        "compare_candidates",
        "stop",
    ]
