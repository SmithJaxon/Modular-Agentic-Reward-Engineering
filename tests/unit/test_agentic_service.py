"""
Summary: Unit tests for autonomous experiment service control loop behavior.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.controller import ControllerContext
from rewardlab.agentic.service import AgentExperimentService
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.orchestrator.session_service import ServicePaths
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.agent_experiment import ActionType, AgentExperimentSpec
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.session_config import EnvironmentBackend


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


@dataclass
class RobustnessThenStopBroker:
    """Broker fake that emits one robustness assessment before stopping."""

    def execute_action(self, **kwargs: object) -> ToolResult:
        """Return deterministic robustness payloads keyed by action type."""

        action = kwargs["action"]
        record = kwargs["record"]
        candidates = kwargs["candidates"]
        action_type = action.action_type.value
        if action_type == "run_robustness_probes":
            candidate_id = candidates[0].candidate_id
            run_id = f"{candidate_id}-robustness-001"
            return ToolResult(
                status="ok",
                summary="robustness complete",
                payload={
                    "candidate_id": candidate_id,
                    "primary_run_id": f"{record.experiment_id}-run-000",
                    "robustness_runs": [
                        {
                            "run_id": run_id,
                            "candidate_id": candidate_id,
                            "backend": record.spec.environment.backend.value,
                            "environment_id": record.spec.environment.id,
                            "run_type": RunType.ROBUSTNESS.value,
                            "execution_mode": ExecutionMode.ACTUAL_BACKEND.value,
                            "variant_label": "seed-17",
                            "status": RunStatus.COMPLETED.value,
                            "metrics": {"episode_reward": 0.4, "train_timesteps": 100},
                            "artifact_refs": ["manifest.json", "metrics.json"],
                            "started_at": datetime(2026, 4, 10, 12, 1, tzinfo=UTC).isoformat(),
                            "ended_at": datetime(2026, 4, 10, 12, 2, tzinfo=UTC).isoformat(),
                        }
                    ],
                    "assessment": {
                        "assessment_id": f"{candidate_id}-robustness",
                        "candidate_id": candidate_id,
                        "backend": record.spec.environment.backend.value,
                        "primary_run_id": f"{record.experiment_id}-run-000",
                        "probe_run_ids": [run_id],
                        "variant_count": 1,
                        "degradation_ratio": 0.52,
                        "risk_level": "high",
                        "risk_notes": "Strong degradation under probe variant.",
                    },
                },
            )
        return ToolResult(status="ok", summary="stop", payload={"stop": True})


@dataclass
class ProposeOnlyBroker:
    """Broker fake that proposes one candidate revision for budget-stop tests."""

    def execute_action(self, **kwargs: object) -> ToolResult:
        """Return a deterministic proposal payload for propose actions."""

        action = kwargs["action"]
        record = kwargs["record"]
        if action.action_type.value != "propose_reward":
            return ToolResult(status="ok", summary="stop", payload={"stop": True})
        candidate_id = f"{record.experiment_id}-candidate-001"
        return ToolResult(
            status="ok",
            summary="candidate proposed",
            payload={
                "candidate": {
                    "candidate_id": candidate_id,
                    "session_id": record.experiment_id,
                    "parent_candidate_id": f"{record.experiment_id}-candidate-000",
                    "iteration_index": 1,
                    "reward_definition": "def reward(observation):\n    return 1.1\n",
                    "change_summary": "budget proposal",
                    "aggregate_score": None,
                    "selected_final": False,
                    "minor_robustness_risk_accepted": False,
                    "created_at": datetime(2026, 4, 10, 12, 1, tzinfo=UTC).isoformat(),
                }
            },
        )


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


def test_reward_generation_budget_stops_loop_when_exhausted() -> None:
    """Service should stop when reward-generation budget is consumed."""

    runtime_root = Path(".agentic-test-runtime-generation-budget")
    paths, repository = _service_for_runtime(runtime_root)
    spec_payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml")
    ).model_dump(mode="python")
    spec_payload["budgets"]["compute"]["max_reward_generations"] = 1
    spec_payload["agent_loop"]["samples_per_iteration"] = 2
    spec_file = runtime_root / "agent_generation_budget.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    controller = SequenceController(
        actions=[
            ControllerAction(
                action_type=ActionType.PROPOSE_REWARD,
                rationale="propose until budget says stop",
                expected_value=0.2,
                expected_cost=0.0,
                action_input={},
            )
        ]
    )
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=controller,  # type: ignore[arg-type]
        tool_broker=ProposeOnlyBroker(),  # type: ignore[arg-type]
    )
    service.initialize()
    result = service.run_experiment(
        spec_file=spec_file,
        experiment_id=f"experiment-generation-budget-{uuid4().hex[:8]}",
    )

    assert result.status.value == "completed"
    assert result.stop_reason == "reward_generation_budget_exhausted"
    status = service.get_status(experiment_id=result.experiment_id)
    assert status.consumed_reward_generations == 1


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


def test_run_robustness_action_persists_assessment_and_probe_runs() -> None:
    """Service should persist robustness runs and assessments from tool results."""

    runtime_root = Path(".agentic-test-runtime-robustness")
    paths, repository = _service_for_runtime(runtime_root)
    spec_payload = load_experiment_spec(
        Path("tools/fixtures/experiments/agent_cartpole_lowcost.yaml")
    ).model_dump(mode="python")
    spec_payload["tool_policy"]["allowed_tools"].append("run_robustness_probes")
    spec_file = runtime_root / "agent_robustness_enabled.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    controller = SequenceController(
        actions=[
            ControllerAction(
                action_type=ActionType.RUN_ROBUSTNESS_PROBES,
                rationale="probe reward-hacking risk",
                expected_value=0.2,
                expected_cost=0.1,
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
        tool_broker=RobustnessThenStopBroker(),  # type: ignore[arg-type]
    )
    service.initialize()

    result = service.run_experiment(
        spec_file=spec_file,
        experiment_id=f"experiment-robustness-{uuid4().hex[:8]}",
    )

    assert result.status.value == "completed"
    trace = service.trace_payload(experiment_id=result.experiment_id)
    assert len(trace["robustness_assessments"]) == 1
    robustness_runs = [
        run for run in trace["runs"] if run["run_type"] == RunType.ROBUSTNESS.value
    ]
    assert len(robustness_runs) == 1
    status = service.get_status(experiment_id=result.experiment_id)
    assert status.consumed_experiments == 1


def test_final_evaluation_runs_after_completion_when_enabled() -> None:
    """Service should execute configured final-evaluation runs after completion."""

    runtime_root = Path(".agentic-test-runtime-final-eval")
    paths, repository = _service_for_runtime(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "version": 1,
        "experiment_name": "final-eval-smoke",
        "objective": "Validate post-loop final evaluation behavior.",
        "environment": {"backend": "gymnasium", "id": "CartPole-v1", "seed": 5},
        "baseline_reward": {
            "mode": "file",
            "path": "tools/fixtures/rewards/cartpole_baseline.py",
            "entrypoint_name": "compute_reward",
        },
        "models": {
            "controller": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 2000,
            },
            "reward_designer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 2000,
            },
            "analyzer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 1000,
            },
        },
        "budgets": {
            "api": {
                "max_total_tokens": 50000,
                "max_total_usd": 5.0,
                "max_completion_tokens_per_call": 4000,
            },
            "time": {"max_wall_clock_minutes": 60},
            "compute": {
                "max_experiments": 20,
                "max_total_train_timesteps": 0,
                "max_reward_generations": 4,
                "max_parallel_experiments": 1,
            },
        },
        "governance": {
            "stopping": {
                "max_iterations": 4,
                "plateau_window": 2,
                "min_relative_improvement": 0.01,
                "max_no_improve_streak": 2,
                "max_failed_actions": 2,
            },
            "human_feedback": {"allow": False, "feedback_gate": "none", "max_requests": 0},
        },
        "tool_policy": {
            "allowed_tools": [
                "run_experiment",
                "propose_reward_revision",
                "summarize_run_artifacts",
                "validate_reward_program",
                "estimate_cost_and_risk",
                "compare_candidates",
                "stop_or_continue_recommendation",
            ],
            "default_timeout_seconds": 300,
            "max_retries_per_tool": 1,
        },
        "agent_loop": {
            "encourage_run_all_after_each_experiment": False,
            "samples_per_iteration": 1,
        },
        "execution": {
            "rollout": {"max_episode_steps": 200},
            "final_evaluation": {
                "enabled": True,
                "num_eval_runs": 3,
                "seed_start": 31,
            },
        },
        "outputs": {
            "runtime_dir": str(runtime_root / "runtime"),
            "report_detail": "full",
            "save_decision_trace": True,
        },
    }
    spec_file = runtime_root / "final_eval_spec.json"
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=FakeController(),
        tool_broker=FakeBroker(),  # type: ignore[arg-type]
    )
    service.initialize()

    result = service.run_experiment(
        spec_file=spec_file,
        experiment_id=f"experiment-final-eval-{uuid4().hex[:8]}",
    )
    assert result.status.value == "completed"
    trace = service.trace_payload(experiment_id=result.experiment_id)
    final_eval_runs = [
        run
        for run in trace["runs"]
        if isinstance(run, dict) and str(run.get("run_id", "")).startswith(
            f"{result.experiment_id}-final-eval-"
        )
    ]
    assert len(final_eval_runs) == 3
    status = service.get_status(experiment_id=result.experiment_id)
    assert status.consumed_experiments == 3
    experiment_payload = trace["experiment"]
    assert experiment_payload["metadata"]["final_eval_completed"] is True


def test_compute_usage_counts_eval_runs_in_train_timestep_ledger() -> None:
    """Service should count total PPO work as train_timesteps * evaluation_run_count."""

    runtime_root = Path(".agentic-test-runtime-compute-ledger")
    paths, repository = _service_for_runtime(runtime_root)
    service = AgentExperimentService(
        paths=paths,
        repository=repository,
        controller=FakeController(),
        tool_broker=FakeBroker(),  # type: ignore[arg-type]
    )
    service.initialize()

    spec_payload = {
        "version": 1,
        "experiment_name": "compute-ledger-smoke",
        "objective": "Validate compute ledger scaling for PPO eval runs.",
        "environment": {"backend": "gymnasium", "id": "Humanoid-v4", "seed": 13},
        "baseline_reward": {
            "mode": "file",
            "path": "tools/fixtures/rewards/humanoid_baseline.py",
            "entrypoint_name": "reward",
        },
        "models": {
            "controller": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 2000,
            },
            "reward_designer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 2000,
            },
            "analyzer": {
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "max_completion_tokens": 1000,
            },
        },
        "budgets": {
            "api": {
                "max_total_tokens": 50000,
                "max_total_usd": 5.0,
                "max_completion_tokens_per_call": 4000,
            },
            "time": {"max_wall_clock_minutes": 60},
            "compute": {
                "max_experiments": 20,
                "max_total_train_timesteps": 0,
                "max_reward_generations": 4,
                "max_parallel_experiments": 1,
            },
        },
        "governance": {
            "stopping": {
                "max_iterations": 4,
                "plateau_window": 2,
                "min_relative_improvement": 0.01,
                "max_no_improve_streak": 2,
                "max_failed_actions": 2,
            },
            "human_feedback": {"allow": False, "feedback_gate": "none", "max_requests": 0},
        },
        "tool_policy": {
            "allowed_tools": [
                "run_experiment",
                "propose_reward_revision",
                "summarize_run_artifacts",
                "validate_reward_program",
                "estimate_cost_and_risk",
                "compare_candidates",
                "stop_or_continue_recommendation",
            ],
            "default_timeout_seconds": 300,
            "max_retries_per_tool": 1,
        },
        "agent_loop": {
            "encourage_run_all_after_each_experiment": False,
            "samples_per_iteration": 1,
        },
        "execution": {
            "ppo": {
                "total_timesteps": 100000,
                "eval_runs": 5,
                "checkpoint_count": 10,
                "eval_episodes_per_checkpoint": 1,
            }
        },
        "outputs": {
            "runtime_dir": str(runtime_root / "runtime"),
            "report_detail": "summary",
            "save_decision_trace": True,
        },
    }
    spec = AgentExperimentSpec.model_validate(spec_payload)
    record = service._start_record(spec=spec, experiment_id=f"experiment-ledger-{uuid4().hex[:8]}")

    run = ExperimentRun(
        run_id=f"{record.experiment_id}-run-001",
        candidate_id=f"{record.experiment_id}-candidate-000",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="Humanoid-v4",
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        seed=13,
        status=RunStatus.COMPLETED,
        metrics={"episode_reward": 0.5, "train_timesteps": 100000, "evaluation_run_count": 5},
        artifact_refs=["manifest.json", "metrics.json"],
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
    )

    updated = service._add_compute_usage(record=record, run=run)

    assert updated.budget_ledger.consumed_experiments == 1
    assert updated.budget_ledger.consumed_train_timesteps == 500000
