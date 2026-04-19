"""
Summary: Service layer for autonomous tool-calling experiments and decision traces.
Created: 2026-04-10
Last Updated: 2026-04-16
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from rewardlab.agentic.benchmarking import (
    BenchmarkRunSummary,
    aggregate_benchmark_summaries,
    summarize_trace_for_benchmark,
)
from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.controller import ControllerAgent, ControllerContext
from rewardlab.agentic.eureka_metrics import (
    compute_eureka_comparison_metrics,
    compute_reward_hacking_metrics,
)
from rewardlab.agentic.policy_engine import PolicyEngine
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.agentic.tool_broker import ToolBroker
from rewardlab.agentic.tools import (
    CompareCandidatesTool,
    EstimateCostAndRiskTool,
    ProposeRewardTool,
    RequestHumanFeedbackTool,
    RunExperimentTool,
    RunRobustnessProbesTool,
    SummarizeRunArtifactsTool,
    ValidateRewardProgramTool,
)
from rewardlab.experiments.artifacts import RunArtifactWriter
from rewardlab.experiments.execution_service import (
    ExecutionRequest,
    ExperimentExecutionService,
    ExperimentRunner,
)
from rewardlab.experiments.reward_program import load_reward_program
from rewardlab.experiments.runner_factory import build_runner
from rewardlab.feedback.human_feedback_service import HumanFeedbackService
from rewardlab.llm.openai_client import OpenAIClient
from rewardlab.orchestrator.reward_designer import (
    DeterministicRewardDesigner,
    OpenAIRewardDesigner,
    RewardDesignConfig,
    RewardDesignerMode,
    RewardDesignRequest,
)
from rewardlab.orchestrator.session_service import (
    ServicePaths,
    resolve_control_mode_from_environment,
)
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.agent_experiment import (
    ActionType,
    AgentBudgetLedger,
    AgentDecisionRecord,
    AgentExperimentRecord,
    AgentExperimentSpec,
    AgentExperimentStatus,
    InitializationMode,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.feedback_entry import FeedbackEntry
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import FeedbackGate

AGENT_EXPERIMENTS_NAMESPACE = "agent_experiments"
AGENT_EXPERIMENT_CANDIDATES_NAMESPACE_PREFIX = "agent_experiment_candidates"
AGENT_EXPERIMENT_DECISIONS_NAMESPACE_PREFIX = "agent_experiment_decisions"
AGENT_EXPERIMENT_FEEDBACK_NAMESPACE_PREFIX = "agent_experiment_feedback"
AGENT_EXPERIMENT_FEEDBACK_REQUESTS_NAMESPACE_PREFIX = "agent_experiment_feedback_requests"


@dataclass(frozen=True)
class StartedAgentExperiment:
    """Response payload returned when starting/running an autonomous experiment."""

    experiment_id: str
    status: AgentExperimentStatus
    stop_reason: str | None
    best_candidate_id: str | None
    report_path: str | None

    def to_json_payload(self) -> dict[str, str | None]:
        """Convert response to a JSON-ready payload."""

        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "best_candidate_id": self.best_candidate_id,
            "report_path": self.report_path,
        }


@dataclass(frozen=True)
class AgentExperimentStatusPayload:
    """Response payload returned by status and stop/resume calls."""

    experiment_id: str
    status: AgentExperimentStatus
    stop_reason: str | None
    best_candidate_id: str | None
    consumed_total_tokens: int
    consumed_total_usd: float
    consumed_experiments: int
    consumed_train_timesteps: int
    consumed_reward_generations: int
    consumed_human_feedback_requests: int

    def to_json_payload(self) -> dict[str, str | int | float | None]:
        """Convert status payload to JSON-safe dictionary."""

        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "best_candidate_id": self.best_candidate_id,
            "consumed_total_tokens": self.consumed_total_tokens,
            "consumed_total_usd": self.consumed_total_usd,
            "consumed_experiments": self.consumed_experiments,
            "consumed_train_timesteps": self.consumed_train_timesteps,
            "consumed_reward_generations": self.consumed_reward_generations,
            "consumed_human_feedback_requests": self.consumed_human_feedback_requests,
        }


@dataclass(frozen=True)
class SubmittedAgentFeedback:
    """Response payload returned when feedback is submitted to an experiment."""

    experiment_id: str
    feedback_id: str
    candidate_id: str
    status: AgentExperimentStatus

    def to_json_payload(self) -> dict[str, str]:
        """Convert feedback response to a JSON-safe payload."""

        return {
            "experiment_id": self.experiment_id,
            "feedback_id": self.feedback_id,
            "candidate_id": self.candidate_id,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class StartedAgentBenchmark:
    """Response payload returned when running a multi-seed benchmark."""

    benchmark_id: str
    run_count: int
    completed_count: int
    improved_count: int
    best_experiment_id: str | None
    best_score: float | None
    report_path: str

    def to_json_payload(self) -> dict[str, str | int | float | None]:
        """Convert benchmark response to a JSON-safe payload."""

        return {
            "benchmark_id": self.benchmark_id,
            "run_count": self.run_count,
            "completed_count": self.completed_count,
            "improved_count": self.improved_count,
            "best_experiment_id": self.best_experiment_id,
            "best_score": self.best_score,
            "report_path": self.report_path,
        }


class AgentExperimentService:
    """Coordinate autonomous tool-calling experiment runs."""

    def __init__(
        self,
        *,
        paths: ServicePaths,
        repository: SessionRepository,
        execution_service: ExperimentExecutionService | None = None,
        controller: ControllerAgent | None = None,
        policy_engine: PolicyEngine | None = None,
        tool_broker: ToolBroker | None = None,
        human_feedback_service: HumanFeedbackService | None = None,
    ) -> None:
        """Create service dependencies for autonomous experiments."""

        self.paths = paths
        self.repository = repository
        self.execution_service = (
            execution_service
            or ExperimentExecutionService(artifact_writer=paths.experiment_artifact_writer())
        )
        self.controller = controller or ControllerAgent()
        self.policy_engine = policy_engine or PolicyEngine()
        self.human_feedback_service = human_feedback_service or HumanFeedbackService()
        self.tool_broker = tool_broker or ToolBroker(
            run_experiment_tool=RunExperimentTool(execution_service=self.execution_service),
            run_robustness_probes_tool=RunRobustnessProbesTool(
                execution_service=self.execution_service
            ),
            propose_reward_tool=ProposeRewardTool(),
            summarize_run_artifacts_tool=SummarizeRunArtifactsTool(),
            validate_reward_program_tool=ValidateRewardProgramTool(),
            estimate_cost_and_risk_tool=EstimateCostAndRiskTool(),
            compare_candidates_tool=CompareCandidatesTool(),
            request_human_feedback_tool=RequestHumanFeedbackTool(),
        )

    @classmethod
    def from_environment(cls) -> AgentExperimentService:
        """Construct a service from the standard runtime environment paths."""

        paths = ServicePaths.from_environment()
        repository = SessionRepository(
            paths=RepositoryPaths(
                database_path=paths.database_path,
                event_log_path=paths.event_log_dir / "events.jsonl",
            )
        )
        service = cls(paths=paths, repository=repository)
        service.initialize()
        return service

    def initialize(self) -> None:
        """Initialize storage directories and persistence backends."""

        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.repository.initialize()
        (self.paths.report_dir / "agent_experiments").mkdir(parents=True, exist_ok=True)

    def validate_spec(self, *, spec_file: Path) -> AgentExperimentSpec:
        """Validate and return a parsed experiment spec."""

        return load_experiment_spec(spec_file)

    def run_experiment(
        self,
        *,
        spec_file: Path,
        experiment_id: str | None = None,
    ) -> StartedAgentExperiment:
        """Start and execute an autonomous experiment loop to completion/pause."""

        spec = load_experiment_spec(spec_file)
        record = self._start_record(
            spec=spec,
            experiment_id=experiment_id,
            spec_file=spec_file,
        )
        record, report_path = self._execute_loop(record=record)
        return StartedAgentExperiment(
            experiment_id=record.experiment_id,
            status=record.status,
            stop_reason=record.stop_reason,
            best_candidate_id=record.best_candidate_id,
            report_path=str(report_path) if report_path is not None else None,
        )

    def run_benchmark(
        self,
        *,
        spec_file: Path,
        seeds: list[int] | None = None,
        benchmark_id: str | None = None,
    ) -> StartedAgentBenchmark:
        """Run a multi-seed benchmark and export an aggregate benchmark report."""

        spec = self.validate_spec(spec_file=spec_file)
        resolved_seeds = (
            sorted(set(seeds))
            if seeds
            else [spec.environment.seed if spec.environment.seed is not None else 0]
        )
        actual_benchmark_id = benchmark_id or _default_benchmark_id()

        runtime_dir = Path(spec.outputs.runtime_dir)
        benchmark_data_dir = runtime_dir / "benchmarks" / actual_benchmark_id
        specs_dir = benchmark_data_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc).isoformat()
        run_summaries: list[BenchmarkRunSummary] = []
        for seed in resolved_seeds:
            seeded_spec = spec.model_copy(
                update={
                    "environment": spec.environment.model_copy(update={"seed": seed})
                }
            )
            seed_spec_path = specs_dir / f"{actual_benchmark_id}-seed-{seed}.json"
            seed_spec_path.write_text(
                json.dumps(seeded_spec.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            experiment_id = _benchmark_experiment_id(actual_benchmark_id, seed)
            self.run_experiment(spec_file=seed_spec_path, experiment_id=experiment_id)
            trace = self.trace_payload(experiment_id=experiment_id)
            run_summaries.append(
                summarize_trace_for_benchmark(
                    seed=seed,
                    trace_payload=trace,
                )
            )

        aggregate = aggregate_benchmark_summaries(run_summaries)
        report_payload: dict[str, object] = {
            "benchmark_id": actual_benchmark_id,
            "spec_file": str(spec_file),
            "seeds": resolved_seeds,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "runs": [item.to_payload() for item in run_summaries],
            "aggregate": aggregate,
        }
        report_dir = runtime_dir / "reports" / "agent_benchmarks"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{actual_benchmark_id}.benchmark.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        overview = aggregate.get("overview")
        best_experiment_id: str | None = None
        best_score: float | None = None
        completed_count = 0
        improved_count = 0
        if isinstance(overview, dict):
            best_value = overview.get("best_experiment_id")
            if isinstance(best_value, str):
                best_experiment_id = best_value
            score_value = overview.get("best_score")
            if isinstance(score_value, (int, float)):
                best_score = float(score_value)
            completed_value = overview.get("completed_count")
            if isinstance(completed_value, int):
                completed_count = completed_value
            improved_value = overview.get("improved_count")
            if isinstance(improved_value, int):
                improved_count = improved_value

        return StartedAgentBenchmark(
            benchmark_id=actual_benchmark_id,
            run_count=len(run_summaries),
            completed_count=completed_count,
            improved_count=improved_count,
            best_experiment_id=best_experiment_id,
            best_score=best_score,
            report_path=str(report_path),
        )

    def _execute_loop(
        self,
        *,
        record: AgentExperimentRecord,
    ) -> tuple[AgentExperimentRecord, Path | None]:
        """Run the autonomous control loop from an existing experiment record."""

        report_path: Path | None = None
        failed_actions = int(record.metadata.get("failed_actions", 0))
        non_progress_actions = int(record.metadata.get("non_progress_actions", 0))
        for _ in range(500):
            candidates = self.list_candidates(record.experiment_id)
            runs = self.list_runs(record.experiment_id)
            no_improve_streak = _no_improve_streak(candidates)

            policy_decision = self.policy_engine.evaluate_stop(
                record=record,
                candidates=candidates,
                failed_actions=failed_actions,
                non_progress_actions=non_progress_actions,
            )
            if policy_decision.should_stop:
                record = self._complete_record(record, reason=policy_decision.reason)
                record = self._run_final_evaluation(record=record)
                report_path = self._write_report(record=record)
                break

            action, controller_tokens = self.controller.choose_action(
                ControllerContext(
                    record=record,
                    candidates=candidates,
                    runs=runs,
                    recent_decisions=[
                        decision.model_dump(mode="json")
                        for decision in self.list_decisions(record.experiment_id)[-5:]
                    ],
                    failed_actions=failed_actions,
                    no_improve_streak=no_improve_streak,
                )
            )
            record = self._add_budget_usage(record, consumed_tokens=controller_tokens)
            if action.action_type == ActionType.STOP:
                stop_block_reason = _controller_stop_block_reason(
                    record=record,
                    candidates=candidates,
                )
                if stop_block_reason is not None:
                    replacement = _required_progress_action(
                        record=record,
                        candidates=candidates,
                        runs=runs,
                        reason=stop_block_reason,
                    )
                    self.repository.append_event(
                        session_id=record.experiment_id,
                        event_type="agent_experiment.stop_blocked",
                        payload={
                            "reason": stop_block_reason,
                            "requested_action_type": action.action_type.value,
                            "replacement_action_type": replacement.action_type.value,
                        },
                    )
                    action = replacement
            broker_action = self._contextualize_action_for_broker(
                record=record,
                action=action,
                candidates=candidates,
                runs=runs,
            )
            tool_result = self.tool_broker.execute_action(
                record=record,
                action=broker_action,
                candidates=candidates,
                runs=runs,
            )
            if tool_result.status != "ok":
                failed_actions += 1
            else:
                failed_actions = 0
            record, progress_made = self._apply_tool_result(
                record=record,
                action=action,
                result=tool_result,
            )
            if tool_result.status == "ok" and not progress_made:
                non_progress_actions += 1
            else:
                non_progress_actions = 0
            record = record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "failed_actions": failed_actions,
                        "non_progress_actions": non_progress_actions,
                    }
                }
            )
            self._save_decision(
                record=record,
                action=action,
                result=tool_result,
                consumed_tokens=controller_tokens + tool_result.consumed_tokens,
                consumed_usd=tool_result.consumed_usd,
            )
            self._save_record(record)

            if action.action_type == ActionType.STOP or bool(tool_result.payload.get("stop")):
                stop_block_reason = _controller_stop_block_reason(
                    record=record,
                    candidates=self.list_candidates(record.experiment_id),
                )
                if stop_block_reason is None:
                    record = self._complete_record(record, reason="controller_stop")
                    record = self._run_final_evaluation(record=record)
                    report_path = self._write_report(record=record)
                    break
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.stop_blocked",
                    payload={
                        "reason": stop_block_reason,
                        "requested_action_type": action.action_type.value,
                        "replacement_action_type": None,
                        "tool_requested_stop": bool(tool_result.payload.get("stop")),
                    },
                )

            if record.status == AgentExperimentStatus.PAUSED:
                report_path = self._write_report(record=record)
                break

            if record.status in {
                AgentExperimentStatus.COMPLETED,
                AgentExperimentStatus.FAILED,
                AgentExperimentStatus.INTERRUPTED,
            }:
                if record.status == AgentExperimentStatus.COMPLETED:
                    record = self._run_final_evaluation(record=record)
                report_path = self._write_report(record=record)
                break
        else:
            record = self._complete_record(record, reason="loop_guard_cap_reached")
            record = self._run_final_evaluation(record=record)
            report_path = self._write_report(record=record)

        self._save_record(record)
        return record, report_path

    def get_status(self, *, experiment_id: str) -> AgentExperimentStatusPayload:
        """Return summarized status for one autonomous experiment."""

        record = self._require_record(experiment_id)
        return AgentExperimentStatusPayload(
            experiment_id=record.experiment_id,
            status=record.status,
            stop_reason=record.stop_reason,
            best_candidate_id=record.best_candidate_id,
            consumed_total_tokens=record.budget_ledger.consumed_total_tokens,
            consumed_total_usd=record.budget_ledger.consumed_total_usd,
            consumed_experiments=record.budget_ledger.consumed_experiments,
            consumed_train_timesteps=record.budget_ledger.consumed_train_timesteps,
            consumed_reward_generations=record.budget_ledger.consumed_reward_generations,
            consumed_human_feedback_requests=(
                record.budget_ledger.consumed_human_feedback_requests
            ),
        )

    def stop_experiment(self, *, experiment_id: str) -> AgentExperimentStatusPayload:
        """Interrupt an experiment and persist an agent report."""

        record = self._require_record(experiment_id)
        if record.status == AgentExperimentStatus.RUNNING:
            record = record.model_copy(
                update={
                    "status": AgentExperimentStatus.INTERRUPTED,
                    "ended_at": datetime.now(timezone.utc),
                    "stop_reason": "user_interrupt",
                }
            )
            self._save_record(record)
            self._write_report(record=record)
        return self.get_status(experiment_id=experiment_id)

    def resume_experiment(self, *, experiment_id: str) -> AgentExperimentStatusPayload:
        """Resume a paused autonomous experiment and continue autonomous looping."""

        record = self._require_record(experiment_id)
        if record.status == AgentExperimentStatus.PAUSED:
            record = record.model_copy(
                update={
                    "status": AgentExperimentStatus.RUNNING,
                    "stop_reason": None,
                }
            )
            self._save_record(record)
            self._execute_loop(record=record)
        return self.get_status(experiment_id=experiment_id)

    def trace_payload(self, *, experiment_id: str) -> dict[str, object]:
        """Return a full trace payload with decisions, candidates, and runs."""

        record = self._require_record(experiment_id)
        return self._build_trace_payload(record=record, for_report=False)

    def _build_trace_payload(
        self,
        *,
        record: AgentExperimentRecord,
        for_report: bool,
    ) -> dict[str, object]:
        """Build either a full or report-shaped trace payload for one experiment."""

        candidates = self.list_candidates(record.experiment_id)
        runs = self.list_runs(record.experiment_id)
        decisions = self.list_decisions(record.experiment_id)
        feedback_requests = self.list_feedback_requests(record.experiment_id)
        feedback_entries = self.list_feedback(record.experiment_id)
        robustness_assessments = self.list_robustness_assessments(record.experiment_id)

        if for_report and record.spec.outputs.report_detail == "summary":
            scored_candidates = [
                candidate for candidate in candidates if candidate.aggregate_score is not None
            ]
            best_score = (
                max(
                    (
                        float(candidate.aggregate_score)
                        for candidate in scored_candidates
                        if candidate.aggregate_score is not None
                    ),
                    default=None,
                )
                if len(scored_candidates) > 0
                else None
            )
            summary_payload: dict[str, object] = {
                "experiment": record.model_dump(mode="json"),
                "summary": {
                    "candidate_count": len(candidates),
                    "run_count": len(runs),
                    "decision_count": len(decisions),
                    "feedback_request_count": len(feedback_requests),
                    "feedback_entry_count": len(feedback_entries),
                    "robustness_assessment_count": len(robustness_assessments),
                    "best_score": best_score,
                    "best_candidate_id": record.best_candidate_id,
                },
            }
            if record.spec.outputs.save_decision_trace:
                summary_payload["recent_decisions"] = [
                    decision.model_dump(mode="json") for decision in decisions[-5:]
                ]
            return summary_payload

        payload: dict[str, object] = {
            "experiment": record.model_dump(mode="json"),
            "candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
            "runs": [run.model_dump(mode="json") for run in runs],
            "feedback_requests": feedback_requests,
            "feedback_entries": [
                feedback.model_dump(mode="json") for feedback in feedback_entries
            ],
            "robustness_assessments": [
                assessment.model_dump(mode="json")
                for assessment in robustness_assessments
            ],
        }
        include_decisions = not for_report or record.spec.outputs.save_decision_trace
        if include_decisions:
            payload["decisions"] = [
                decision.model_dump(mode="json") for decision in decisions
            ]
        return payload

    def submit_human_feedback(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        comment: str,
        score: float | None = None,
        request_id: str | None = None,
        artifact_ref: str | None = None,
    ) -> SubmittedAgentFeedback:
        """Persist one human feedback entry and unblock paused feedback gates."""

        record = self._require_record(experiment_id)
        if not record.spec.governance.human_feedback.allow:
            raise ValueError("human feedback is disabled for this experiment")

        candidate_ids = {
            candidate.candidate_id for candidate in self.list_candidates(experiment_id)
        }
        if candidate_id not in candidate_ids:
            raise ValueError(f"candidate {candidate_id!r} does not exist")

        pending_requests = self.list_feedback_requests(experiment_id)
        if request_id is not None:
            matched_request = next(
                (
                    item
                    for item in pending_requests
                    if item.get("request_id") == request_id
                ),
                None,
            )
            if matched_request is None:
                raise ValueError(f"feedback request {request_id!r} does not exist")
            requested_candidate_id = matched_request.get("candidate_id")
            if requested_candidate_id != candidate_id:
                raise ValueError(
                    "feedback candidate_id must match the candidate referenced by request_id"
                )

        feedback = self.human_feedback_service.submit_feedback(
            session_id=experiment_id,
            candidate_id=candidate_id,
            comment=comment,
            score=score,
            artifact_ref=artifact_ref,
        )
        self._save_feedback(experiment_id=experiment_id, feedback=feedback)
        self.repository.append_event(
            session_id=experiment_id,
            event_type="agent_experiment.feedback_submitted",
            payload={
                "feedback_id": feedback.feedback_id,
                "candidate_id": feedback.candidate_id,
                "request_id": request_id,
            },
        )

        if record.status == AgentExperimentStatus.PAUSED and (
            record.stop_reason == "awaiting_human_feedback"
        ):
            metadata = dict(record.metadata)
            metadata.pop("awaiting_feedback_request_id", None)
            if request_id is not None:
                metadata["last_feedback_request_id"] = request_id
            metadata["last_feedback_id"] = feedback.feedback_id
            updated = record.model_copy(update={"metadata": metadata})
            self._save_record(updated)
            record = updated

        return SubmittedAgentFeedback(
            experiment_id=experiment_id,
            feedback_id=feedback.feedback_id,
            candidate_id=candidate_id,
            status=record.status,
        )

    def list_candidates(self, experiment_id: str) -> list[RewardCandidate]:
        """Return experiment candidates ordered by iteration index."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _candidate_namespace(experiment_id)
        )
        candidates = [RewardCandidate.model_validate(payload) for payload in payloads]
        return sorted(candidates, key=lambda candidate: candidate.iteration_index)

    def list_runs(self, experiment_id: str) -> list[ExperimentRun]:
        """Return experiment runs linked to the experiment candidate lineage."""

        candidate_ids = {
            candidate.candidate_id for candidate in self.list_candidates(experiment_id)
        }
        session_prefix = f"{experiment_id}-"
        runs = self.repository.list_experiment_runs()
        filtered = [
            run
            for run in runs
            if run.candidate_id in candidate_ids or run.run_id.startswith(session_prefix)
        ]
        return sorted(filtered, key=lambda run: (run.started_at or run.ended_at, run.run_id))

    def list_decisions(self, experiment_id: str) -> list[AgentDecisionRecord]:
        """Return persisted decision trace entries ordered by step index."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _decision_namespace(experiment_id)
        )
        decisions = [AgentDecisionRecord.model_validate(payload) for payload in payloads]
        return sorted(decisions, key=lambda decision: (decision.step_index, decision.created_at))

    def list_feedback(self, experiment_id: str) -> list[FeedbackEntry]:
        """Return persisted human feedback entries ordered by creation time."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _feedback_namespace(experiment_id)
        )
        entries = [FeedbackEntry.model_validate(payload) for payload in payloads]
        return sorted(entries, key=lambda item: item.created_at)

    def list_feedback_requests(self, experiment_id: str) -> list[dict[str, object]]:
        """Return stored feedback request envelopes for one experiment."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _feedback_request_namespace(experiment_id)
        )
        normalized: list[dict[str, object]] = []
        for payload in payloads:
            if isinstance(payload, dict):
                normalized.append(dict(payload))
        return sorted(
            normalized,
            key=lambda item: str(item.get("created_at", "")),
        )

    def list_robustness_assessments(
        self,
        experiment_id: str,
    ) -> list[RobustnessAssessment]:
        """Return robustness assessments for candidates in one experiment."""

        candidate_ids = {
            candidate.candidate_id for candidate in self.list_candidates(experiment_id)
        }
        assessments = self.repository.list_robustness_assessments()
        filtered = [
            assessment
            for assessment in assessments
            if assessment.candidate_id in candidate_ids
        ]
        return sorted(
            filtered,
            key=lambda item: (item.created_at, item.assessment_id),
        )

    def _start_record(
        self,
        *,
        spec: AgentExperimentSpec,
        experiment_id: str | None,
        spec_file: Path | None = None,
    ) -> AgentExperimentRecord:
        """Create and persist a new running experiment with baseline candidate."""

        actual_id = experiment_id or _default_experiment_id()
        existing = self._get_record(actual_id)
        if existing is not None:
            return existing

        init_mode = spec.initialization.mode
        seed_candidates: list[RewardCandidate]
        if init_mode == InitializationMode.HUMAN:
            baseline_path = Path(spec.baseline_reward.path)
            baseline_source = baseline_path.read_text(encoding="utf-8").strip()
            if not baseline_source:
                raise ValueError("baseline reward source is blank")
            seed_candidates = [
                RewardCandidate(
                    candidate_id=f"{actual_id}-candidate-000",
                    session_id=actual_id,
                    iteration_index=0,
                    reward_definition=baseline_source,
                    change_summary="Baseline candidate loaded from experiment spec reward file.",
                    aggregate_score=None,
                )
            ]
        else:
            seed_candidates = self._build_default_init_seed_candidates(
                spec=spec,
                experiment_id=actual_id,
            )
            if len(seed_candidates) == 0:
                raise ValueError("default initialization did not produce any seed candidates")

        now = datetime.now(timezone.utc)
        seed_candidate_count = len(seed_candidates)
        record = AgentExperimentRecord(
            experiment_id=actual_id,
            status=AgentExperimentStatus.RUNNING,
            spec=spec,
            created_at=now,
            started_at=now,
            best_candidate_id=seed_candidates[0].candidate_id,
            metadata={
                "failed_actions": 0,
                "non_progress_actions": 0,
                "control_mode": resolve_control_mode_from_environment().value,
                "runtime_dir": spec.outputs.runtime_dir,
                "init_mode": init_mode.value,
                "init_seed_candidate_count": seed_candidate_count,
                "spec_file_stem": (
                    spec_file.stem if spec_file is not None else spec.experiment_name
                ),
            },
            budget_ledger=AgentBudgetLedger(),
        )
        self._save_record(record)
        for candidate in seed_candidates:
            self._save_candidate(candidate)
        self.repository.append_event(
            session_id=actual_id,
            event_type="agent_experiment.started",
            payload={
                "baseline_candidate_id": seed_candidates[0].candidate_id,
                "seed_candidate_count": seed_candidate_count,
                "environment_id": spec.environment.id,
                "init_mode": init_mode.value,
            },
        )
        return record

    def _build_default_init_seed_candidates(
        self,
        *,
        spec: AgentExperimentSpec,
        experiment_id: str,
    ) -> list[RewardCandidate]:
        """Generate default-init seed candidates using the reward designer stack."""

        entrypoint_name = spec.baseline_reward.entrypoint_name
        seed_template = _default_initial_reward_source(entrypoint_name=entrypoint_name)
        seed_count = spec.initialization.default_seed_candidate_count or 1
        seed_count = max(seed_count, 1)

        fallback_designer = DeterministicRewardDesigner()
        model_cfg = spec.models.reward_designer
        openai_client = OpenAIClient()
        if openai_client.has_credentials:
            designer: OpenAIRewardDesigner | DeterministicRewardDesigner = OpenAIRewardDesigner(
                openai_client=openai_client,
                config=RewardDesignConfig(
                    mode=RewardDesignerMode.OPENAI,
                    model=model_cfg.model,
                    reasoning_effort=_coerce_reasoning_effort(model_cfg.reasoning_effort),
                    max_tokens=min(
                        model_cfg.max_completion_tokens,
                        spec.budgets.api.max_completion_tokens_per_call,
                    ),
                ),
            )
            designer_label = "openai"
        else:
            designer = fallback_designer
            designer_label = "deterministic"

        allowed_parameter_names = _resolve_default_init_allowed_parameter_names(
            spec=spec,
            seed_template_source=seed_template,
        )
        anchor_candidate = RewardCandidate(
            candidate_id=f"{experiment_id}-candidate-anchor",
            session_id=experiment_id,
            iteration_index=0,
            reward_definition=seed_template,
            change_summary="Default-init internal anchor candidate.",
            aggregate_score=None,
        )
        seeds: list[RewardCandidate] = []
        for index in range(seed_count):
            sample_number = index + 1
            request = RewardDesignRequest(
                session_id=experiment_id,
                objective_text=spec.objective,
                environment_id=spec.environment.id,
                environment_backend=spec.environment.backend,
                current_candidate=anchor_candidate,
                next_iteration_index=index,
                prior_candidates=tuple([anchor_candidate, *seeds]),
                allowed_parameter_names=allowed_parameter_names,
            )

            reward_definition = seed_template
            mode_summary = f"{designer_label}_fallback_template"
            try:
                design_result = designer.design_next_candidate(request)
                reward_definition = design_result.reward_definition.strip()
                mode_summary = designer_label
            except Exception:
                try:
                    design_result = fallback_designer.design_next_candidate(request)
                    reward_definition = design_result.reward_definition.strip()
                    mode_summary = "deterministic_fallback"
                except Exception:
                    reward_definition = seed_template
                    mode_summary = "template_fallback"

            if not reward_definition:
                reward_definition = seed_template
                mode_summary = "template_fallback"

            seeds.append(
                RewardCandidate(
                    candidate_id=f"{experiment_id}-candidate-{index:03d}",
                    session_id=experiment_id,
                    iteration_index=0,
                    reward_definition=reward_definition,
                    change_summary=(
                        "Eureka-style default-init bootstrap seed "
                        f"({sample_number}/{seed_count}, source={mode_summary})."
                    ),
                    aggregate_score=None,
                )
            )
        return seeds

    def _apply_tool_result(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        result: ToolResult,
    ) -> tuple[AgentExperimentRecord, bool]:
        """Apply a tool result to experiment state and persistence."""

        record = self._add_budget_usage(
            record,
            consumed_tokens=result.consumed_tokens,
            consumed_usd=result.consumed_usd,
        )

        if action.action_type == ActionType.RUN_EXPERIMENT:
            run_payloads = result.payload.get("runs")
            if isinstance(run_payloads, list):
                parsed_runs = [
                    ExperimentRun.model_validate(item)
                    for item in run_payloads
                    if isinstance(item, dict)
                ]
            else:
                single_run_payload = result.payload.get("run")
                parsed_runs = (
                    [ExperimentRun.model_validate(single_run_payload)]
                    if isinstance(single_run_payload, dict)
                    else []
                )

            candidate_payloads = result.payload.get("candidates")
            if isinstance(candidate_payloads, list):
                updated_candidates = {
                    candidate.candidate_id: candidate
                    for candidate in (
                        RewardCandidate.model_validate(item)
                        for item in candidate_payloads
                        if isinstance(item, dict)
                    )
                }
            else:
                single_candidate_payload = result.payload.get("candidate")
                single_candidate = (
                    RewardCandidate.model_validate(single_candidate_payload)
                    if isinstance(single_candidate_payload, dict)
                    else None
                )
                updated_candidates = (
                    {single_candidate.candidate_id: single_candidate}
                    if single_candidate is not None
                    else {}
                )

            completed_runs = 0
            for run in parsed_runs:
                self.repository.save_experiment_run(run)
                if run.status == RunStatus.COMPLETED:
                    updated_candidate = updated_candidates.get(run.candidate_id)
                    if updated_candidate is not None:
                        self._save_candidate(updated_candidate)
                    record = self._add_compute_usage(record=record, run=run)
                    completed_runs += 1
                    self.repository.append_event(
                        session_id=record.experiment_id,
                        event_type="agent_experiment.run_completed",
                        payload={
                            "run_id": run.run_id,
                            "candidate_id": run.candidate_id,
                            "score": (
                                updated_candidate.aggregate_score
                                if updated_candidate is not None
                                else None
                            ),
                        },
                    )
                    continue
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.run_failed",
                    payload={
                        "run_id": run.run_id,
                        "candidate_id": run.candidate_id,
                        "failure_reason": run.failure_reason,
                    },
                )

            if completed_runs > 0:
                best = _best_candidate(self.list_candidates(record.experiment_id))
                record = record.model_copy(update={"best_candidate_id": best.candidate_id})

            if result.status != "ok":
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.tool_error",
                    payload={
                        "action": action.action_type.value,
                        "summary": result.summary,
                    },
                )
            return record, completed_runs > 0

        if result.status != "ok":
            self.repository.append_event(
                session_id=record.experiment_id,
                event_type="agent_experiment.tool_error",
                payload={
                    "action": action.action_type.value,
                    "summary": result.summary,
                },
            )
            return record, False

        if action.action_type == ActionType.PROPOSE_REWARD:
            candidate = RewardCandidate.model_validate(result.payload["candidate"])
            self._save_candidate(candidate)
            record = self._add_reward_generation_usage(record)
            self.repository.append_event(
                session_id=record.experiment_id,
                event_type="agent_experiment.candidate_proposed",
                payload={"candidate_id": candidate.candidate_id},
            )
            return record, True

        if action.action_type == ActionType.RUN_ROBUSTNESS_PROBES:
            run_payloads = result.payload.get("robustness_runs")
            robustness_runs = (
                [ExperimentRun.model_validate(item) for item in run_payloads]
                if isinstance(run_payloads, list)
                else []
            )
            for run in robustness_runs:
                self.repository.save_experiment_run(run)
                is_completed = run.status.value == "completed"
                if is_completed:
                    record = self._add_compute_usage(record=record, run=run)
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type=(
                        "agent_experiment.robustness_run_completed"
                        if is_completed
                        else "agent_experiment.robustness_run_failed"
                    ),
                    payload={
                        "run_id": run.run_id,
                        "candidate_id": run.candidate_id,
                        "variant_label": run.variant_label,
                        "status": run.status.value,
                        "artifact_refs": run.artifact_refs,
                        "failure_reason": run.failure_reason,
                    },
                )
            assessment_payload = result.payload.get("assessment")
            if isinstance(assessment_payload, dict):
                assessment = RobustnessAssessment.model_validate(assessment_payload)
                self.repository.save_robustness_assessment(assessment)
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.robustness_assessed",
                    payload={
                        "assessment_id": assessment.assessment_id,
                        "candidate_id": assessment.candidate_id,
                        "risk_level": assessment.risk_level.value,
                        "degradation_ratio": assessment.degradation_ratio,
                    },
                )
            return record, len(robustness_runs) > 0

        if action.action_type == ActionType.REQUEST_HUMAN_FEEDBACK:
            request_id = result.payload.get("request_id")
            created_at = datetime.now(timezone.utc).isoformat()
            request_key = (
                request_id
                if isinstance(request_id, str) and request_id
                else f"{record.experiment_id}-feedback-request-{uuid4().hex[:10]}"
            )
            request_payload = {
                "request_id": request_key,
                "candidate_id": result.payload.get("candidate_id"),
                "prompt": result.payload.get("prompt"),
                "feedback_gate": result.payload.get("feedback_gate"),
                "created_at": created_at,
            }
            self._save_feedback_request(
                experiment_id=record.experiment_id,
                request_id=request_key,
                payload=request_payload,
            )
            record = self._add_human_feedback_usage(record)
            self.repository.append_event(
                session_id=record.experiment_id,
                event_type="agent_experiment.feedback_requested",
                payload={
                    "request_id": request_key,
                    "candidate_id": result.payload.get("candidate_id"),
                },
            )
            if record.spec.governance.human_feedback.feedback_gate != FeedbackGate.NONE:
                metadata = {
                    **record.metadata,
                    "awaiting_feedback_request_id": request_key,
                }
                record = record.model_copy(
                    update={
                        "status": AgentExperimentStatus.PAUSED,
                        "stop_reason": "awaiting_human_feedback",
                        "metadata": metadata,
                    }
                )
            return record, False

        if action.action_type in {
            ActionType.SUMMARIZE_RUN_ARTIFACTS,
            ActionType.VALIDATE_REWARD_PROGRAM,
            ActionType.ESTIMATE_COST_AND_RISK,
            ActionType.COMPARE_CANDIDATES,
            ActionType.STOP,
        }:
            self.repository.append_event(
                session_id=record.experiment_id,
                event_type="agent_experiment.analysis_completed",
                payload={
                    "action": action.action_type.value,
                    "summary": result.summary,
                },
            )
            return record, False

        return record, False

    def _add_budget_usage(
        self,
        record: AgentExperimentRecord,
        *,
        consumed_tokens: int = 0,
        consumed_usd: float = 0.0,
    ) -> AgentExperimentRecord:
        """Increment token/USD budget usage in the record ledger."""

        ledger = record.budget_ledger.model_copy(
            update={
                "consumed_total_tokens": (
                    record.budget_ledger.consumed_total_tokens + max(consumed_tokens, 0)
                ),
                "consumed_total_usd": (
                    record.budget_ledger.consumed_total_usd + max(consumed_usd, 0.0)
                ),
            }
        )
        return record.model_copy(update={"budget_ledger": ledger})

    def _add_compute_usage(
        self,
        *,
        record: AgentExperimentRecord,
        run: ExperimentRun,
    ) -> AgentExperimentRecord:
        """Increment compute budget usage from a completed run."""

        train_timesteps = int(run.metrics.get("train_timesteps", 0))
        evaluation_run_count = int(run.metrics.get("evaluation_run_count", 1))
        if evaluation_run_count < 1:
            evaluation_run_count = 1
        effective_train_timesteps = max(train_timesteps, 0) * evaluation_run_count
        ledger = record.budget_ledger.model_copy(
            update={
                "consumed_experiments": record.budget_ledger.consumed_experiments + 1,
                "consumed_train_timesteps": (
                    record.budget_ledger.consumed_train_timesteps + effective_train_timesteps
                ),
            }
        )
        return record.model_copy(update={"budget_ledger": ledger})

    def _add_human_feedback_usage(
        self,
        record: AgentExperimentRecord,
    ) -> AgentExperimentRecord:
        """Increment the human-feedback request counter in the budget ledger."""

        ledger = record.budget_ledger.model_copy(
            update={
                "consumed_human_feedback_requests": (
                    record.budget_ledger.consumed_human_feedback_requests + 1
                )
            }
        )
        return record.model_copy(update={"budget_ledger": ledger})

    def _add_reward_generation_usage(
        self,
        record: AgentExperimentRecord,
    ) -> AgentExperimentRecord:
        """Increment reward-generation usage in the budget ledger."""

        ledger = record.budget_ledger.model_copy(
            update={
                "consumed_reward_generations": (
                    record.budget_ledger.consumed_reward_generations + 1
                )
            }
        )
        return record.model_copy(update={"budget_ledger": ledger})

    def _contextualize_action_for_broker(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
    ) -> ControllerAction:
        """Attach curated context to run/propose actions before tool execution."""

        if action.action_type == ActionType.RUN_EXPERIMENT:
            return _contextualize_run_action_for_parallel_dispatch(
                record=record,
                action=action,
                candidates=candidates,
                runs=runs,
            )

        if action.action_type != ActionType.PROPOSE_REWARD:
            return action

        experiment_id = record.experiment_id
        recent_decisions = [
            {
                "action_type": decision.action_type.value,
                "rationale": decision.rationale,
                "result_status": decision.result_status,
                "result_summary": decision.result_summary,
            }
            for decision in self.list_decisions(experiment_id)[-8:]
        ]
        recent_feedback = [
            {
                "candidate_id": feedback.candidate_id,
                "comment": feedback.comment,
                "score": feedback.score,
                "created_at": feedback.created_at.isoformat(),
            }
            for feedback in self.list_feedback(experiment_id)[-5:]
        ]
        recent_robustness = [
            {
                "candidate_id": assessment.candidate_id,
                "risk_level": assessment.risk_level.value,
                "degradation_ratio": assessment.degradation_ratio,
                "risk_notes": assessment.risk_notes,
                "created_at": assessment.created_at.isoformat(),
            }
            for assessment in self.list_robustness_assessments(experiment_id)[-5:]
        ]
        candidate_history = [
            {
                "candidate_id": candidate.candidate_id,
                "iteration_index": candidate.iteration_index,
                "aggregate_score": candidate.aggregate_score,
                "change_summary": candidate.change_summary,
            }
            for candidate in sorted(candidates, key=lambda item: item.iteration_index)[-12:]
        ]
        contextual_input = {
            **action.action_input,
            "recent_decision_context": recent_decisions,
            "recent_feedback_context": recent_feedback,
            "recent_robustness_context": recent_robustness,
            "candidate_history_context": candidate_history,
        }
        return action.model_copy(update={"action_input": contextual_input})

    def _save_decision(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        result: ToolResult,
        consumed_tokens: int,
        consumed_usd: float,
    ) -> None:
        """Persist one decision-trace record."""

        step_index = len(self.list_decisions(record.experiment_id))
        decision = AgentDecisionRecord(
            decision_id=f"{record.experiment_id}-decision-{step_index:03d}",
            experiment_id=record.experiment_id,
            step_index=step_index,
            action_type=action.action_type,
            rationale=action.rationale,
            expected_value=action.expected_value,
            expected_cost=action.expected_cost,
            action_input=action.action_input,
            result_status=result.status,
            result_summary=result.summary,
            result_payload=result.payload,
            consumed_tokens=consumed_tokens,
            consumed_usd=consumed_usd,
        )
        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_decision_namespace(record.experiment_id),
            item_key=decision.decision_id,
            payload=decision.model_dump(mode="json"),
            updated_at=decision.created_at.isoformat(),
        )

    def _save_candidate(self, candidate: RewardCandidate) -> None:
        """Persist one experiment candidate."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_candidate_namespace(candidate.session_id),
            item_key=candidate.candidate_id,
            payload=candidate.model_dump(mode="json"),
            updated_at=candidate.created_at.isoformat(),
        )

    def _save_feedback(self, *, experiment_id: str, feedback: FeedbackEntry) -> None:
        """Persist one feedback entry for an experiment."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_feedback_namespace(experiment_id),
            item_key=feedback.feedback_id,
            payload=feedback.model_dump(mode="json"),
            updated_at=feedback.created_at.isoformat(),
        )

    def _save_feedback_request(
        self,
        *,
        experiment_id: str,
        request_id: str,
        payload: dict[str, object],
    ) -> None:
        """Persist one feedback request envelope for an experiment."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_feedback_request_namespace(experiment_id),
            item_key=request_id,
            payload=payload,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_final_evaluation(
        self,
        *,
        record: AgentExperimentRecord,
    ) -> AgentExperimentRecord:
        """Optionally run a post-loop multi-seed final evaluation for the best candidate."""

        final_cfg = record.spec.execution.final_evaluation
        if not final_cfg.enabled:
            return record
        if bool(record.metadata.get("final_eval_completed", False)):
            return record

        candidates = self.list_candidates(record.experiment_id)
        if len(candidates) == 0:
            return record
        best_candidate = _resolve_best_candidate_for_final_eval(
            record=record,
            candidates=candidates,
        )
        if best_candidate is None:
            return record

        spec = record.spec
        rollout = spec.execution.rollout
        base_seed = (
            spec.environment.seed
            if spec.environment.seed is not None
            else final_cfg.seed_start
        )
        execution_service = _resolve_execution_service(
            default_service=self.execution_service,
            runtime_dir=Path(spec.outputs.runtime_dir),
        )
        runner = _build_runner_for_spec(
            spec=spec,
            total_timesteps_override=final_cfg.total_timesteps_override,
            eval_runs_override=final_cfg.eval_runs_override,
        )

        completed_scores: list[float] = []
        completed_count = 0
        failed_count = 0
        for run_index in range(final_cfg.num_eval_runs):
            eval_seed = base_seed + run_index
            run_id = f"{record.experiment_id}-final-eval-{run_index + 1:03d}"
            request = ExecutionRequest(
                run_id=run_id,
                backend=spec.environment.backend,
                environment_id=spec.environment.id,
                execution_mode=ExecutionMode.ACTUAL_BACKEND,
                variant_label=f"final_eval_seed_{eval_seed}",
                seed=eval_seed,
                entrypoint_name=spec.baseline_reward.entrypoint_name,
                max_episode_steps=(
                    rollout.max_episode_steps if rollout is not None else None
                ),
            )
            result = execution_service.execute_candidate(
                candidate=best_candidate,
                request=request,
                runner=runner,
            )
            run = result.run
            self.repository.save_experiment_run(run)
            if run.status == RunStatus.COMPLETED:
                completed_count += 1
                record = self._add_compute_usage(record=record, run=run)
                score = _score_from_run_metrics(run.metrics)
                if score is not None:
                    completed_scores.append(score)
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.final_evaluation_run_completed",
                    payload={
                        "run_id": run.run_id,
                        "candidate_id": run.candidate_id,
                        "variant_label": run.variant_label,
                        "score": score,
                        "artifact_refs": run.artifact_refs,
                    },
                )
            else:
                failed_count += 1
                self.repository.append_event(
                    session_id=record.experiment_id,
                    event_type="agent_experiment.final_evaluation_run_failed",
                    payload={
                        "run_id": run.run_id,
                        "candidate_id": run.candidate_id,
                        "variant_label": run.variant_label,
                        "failure_reason": run.failure_reason,
                    },
                )

        final_eval_mean = (
            (sum(completed_scores) / len(completed_scores))
            if len(completed_scores) > 0
            else None
        )
        metadata: dict[str, str | int | float | bool] = {
            **record.metadata,
            "final_eval_completed": True,
            "final_eval_candidate_id": best_candidate.candidate_id,
            "final_eval_num_runs": final_cfg.num_eval_runs,
            "final_eval_completed_runs": completed_count,
            "final_eval_failed_runs": failed_count,
        }
        if final_eval_mean is not None:
            metadata["final_eval_mean_score"] = round(final_eval_mean, 6)
        record = record.model_copy(update={"metadata": metadata})
        self.repository.append_event(
            session_id=record.experiment_id,
            event_type="agent_experiment.final_evaluation_completed",
            payload={
                "candidate_id": best_candidate.candidate_id,
                "requested_runs": final_cfg.num_eval_runs,
                "completed_runs": completed_count,
                "failed_runs": failed_count,
                "mean_score": final_eval_mean,
            },
        )
        return record

    def _build_comparison_metrics_payload(
        self,
        *,
        record: AgentExperimentRecord,
    ) -> tuple[dict[str, object], AgentExperimentRecord]:
        """Build/ensure Eureka-style comparison metrics and probe-based hacking metrics."""

        comparison_cfg = record.spec.execution.comparison
        base_payload: dict[str, object] = {
            "status": "disabled" if not comparison_cfg.enabled else "unavailable",
            "init_mode": record.spec.initialization.mode.value,
            "hns": None,
            "reward_hacking": None,
        }
        if not comparison_cfg.enabled:
            return base_payload, record

        if record.status != AgentExperimentStatus.COMPLETED:
            return {
                **base_payload,
                "reason": "comparison metrics run only for completed experiments",
            }, record

        candidates = self.list_candidates(record.experiment_id)
        if len(candidates) == 0:
            return {**base_payload, "reason": "no candidates available"}, record
        best_candidate = _resolve_best_candidate_for_final_eval(
            record=record,
            candidates=candidates,
        )
        if best_candidate is None:
            return {**base_payload, "reason": "no best candidate available"}, record
        runs = self.list_runs(record.experiment_id)
        has_performance_evidence = any(
            run.candidate_id == best_candidate.candidate_id
            and run.run_type == RunType.PERFORMANCE
            and run.status == RunStatus.COMPLETED
            for run in runs
        )
        if not has_performance_evidence:
            return {
                **base_payload,
                "reason": "no completed performance run found for best candidate",
            }, record

        resolved_human_path = _resolve_human_reward_path(record=record)
        resolved_sparse_path = _resolve_sparse_reward_path(record=record)
        if resolved_human_path is None or not resolved_human_path.exists():
            return {
                **base_payload,
                "reason": "human baseline reward path is missing or does not exist",
            }, record
        if resolved_sparse_path is None or not resolved_sparse_path.exists():
            return {
                **base_payload,
                "reason": "sparse baseline reward path is missing or does not exist",
            }, record

        spec = record.spec
        rollout = spec.execution.rollout
        execution_service = _resolve_execution_service(
            default_service=self.execution_service,
            runtime_dir=Path(spec.outputs.runtime_dir),
        )
        runner = _build_runner_for_spec(
            spec=spec,
            total_timesteps_override=comparison_cfg.total_timesteps_override,
            eval_runs_override=comparison_cfg.eval_runs_override,
        )
        run_cache = {run.run_id: run for run in runs}
        baseline_entrypoint = spec.baseline_reward.entrypoint_name
        comparison_entrypoint = comparison_cfg.entrypoint_name
        max_episode_steps = rollout.max_episode_steps if rollout is not None else None

        human_candidate = RewardCandidate(
            candidate_id=f"{record.experiment_id}-comparison-human",
            session_id=record.experiment_id,
            iteration_index=0,
            reward_definition=resolved_human_path.read_text(encoding="utf-8"),
            change_summary="Human baseline reward used for HNS comparison.",
            aggregate_score=None,
        )
        sparse_candidate = RewardCandidate(
            candidate_id=f"{record.experiment_id}-comparison-sparse",
            session_id=record.experiment_id,
            iteration_index=0,
            reward_definition=resolved_sparse_path.read_text(encoding="utf-8"),
            change_summary="Sparse baseline reward used for HNS comparison.",
            aggregate_score=None,
        )
        score_blocks: dict[str, dict[str, object]] = {}
        record, score_blocks["method"] = self._ensure_comparison_run_block(
            record=record,
            run_cache=run_cache,
            execution_service=execution_service,
            runner=runner,
            candidate=best_candidate,
            label="method",
            run_count=comparison_cfg.num_eval_runs,
            seed_start=comparison_cfg.seed_start,
            entrypoint_name=baseline_entrypoint,
            max_episode_steps=max_episode_steps,
        )
        for label, candidate in (("human", human_candidate), ("sparse", sparse_candidate)):
            record, score_blocks[label] = self._ensure_comparison_run_block(
                record=record,
                run_cache=run_cache,
                execution_service=execution_service,
                runner=runner,
                candidate=candidate,
                label=label,
                run_count=comparison_cfg.num_eval_runs,
                seed_start=comparison_cfg.seed_start,
                entrypoint_name=comparison_entrypoint,
                max_episode_steps=max_episode_steps,
            )

        method_scores = _score_list_from_block(score_blocks["method"])
        human_scores = _score_list_from_block(score_blocks["human"])
        sparse_scores = _score_list_from_block(score_blocks["sparse"])
        if len(method_scores) == 0 or len(human_scores) == 0 or len(sparse_scores) == 0:
            return {
                **base_payload,
                "reason": "comparison runs completed with insufficient score evidence",
                "scores": score_blocks,
            }, record

        method_score = sum(method_scores) / len(method_scores)
        human_score = sum(human_scores) / len(human_scores)
        sparse_score = sum(sparse_scores) / len(sparse_scores)
        try:
            hns = compute_eureka_comparison_metrics(
                method_score=method_score,
                human_score=human_score,
                sparse_score=sparse_score,
                method_score_source="comparison.method.mean_score",
                human_score_source="comparison.human.mean_score",
                sparse_score_source="comparison.sparse.mean_score",
            )
        except ValueError as exc:
            return {
                **base_payload,
                "reason": str(exc),
                "scores": score_blocks,
            }, record

        record, probe_block = self._ensure_comparison_run_block(
            record=record,
            run_cache=run_cache,
            execution_service=execution_service,
            runner=runner,
            candidate=best_candidate,
            label="probe",
            run_count=comparison_cfg.probe_run_count,
            seed_start=comparison_cfg.probe_seed_start,
            entrypoint_name=baseline_entrypoint,
            max_episode_steps=max_episode_steps,
        )
        probe_scores = _score_list_from_block(probe_block)
        hacking_payload: dict[str, object]
        if len(probe_scores) > 0:
            hacking = compute_reward_hacking_metrics(
                method_score=method_score,
                human_score=human_score,
                sparse_score=sparse_score,
                probe_scores=probe_scores,
            )
            hacking_payload = {
                "probe_runs": probe_block,
                "probe_count": hacking.probe_count,
                "probe_min_score": hacking.probe_min_score,
                "probe_mean_score": hacking.probe_mean_score,
                "probe_below_sparse_rate": hacking.probe_below_sparse_rate,
                "worst_score_degradation_ratio": hacking.worst_score_degradation_ratio,
                "mean_score_degradation_ratio": hacking.mean_score_degradation_ratio,
                "probe_min_human_normalized_score": hacking.probe_min_human_normalized_score,
                "probe_mean_human_normalized_score": hacking.probe_mean_human_normalized_score,
                "worst_human_normalized_drop": hacking.worst_human_normalized_drop,
                "mean_human_normalized_drop": hacking.mean_human_normalized_drop,
                "perils_relative_reward_function_performance": (
                    hacking.perils_relative_reward_function_performance
                ),
                "perils_hacking_severity": hacking.perils_hacking_severity,
                "hacking_risk_index": hacking.hacking_risk_index,
                "hacking_risk_level": hacking.hacking_risk_level,
            }
        else:
            hacking_payload = {
                "status": "unavailable",
                "reason": "probe runs did not produce any completed scores",
                "probe_runs": probe_block,
            }

        return (
            {
                "status": "ok",
                "init_mode": record.spec.initialization.mode.value,
                "scores": score_blocks,
                "hns": {
                    "method_score": hns.method_score,
                    "human_score": hns.human_score,
                    "sparse_score": hns.sparse_score,
                    "human_normalized_score": hns.human_normalized_score,
                    "human_normalized_score_clipped": hns.human_normalized_score_clipped,
                    "delta_vs_human_score": hns.delta_vs_human_score,
                    "delta_vs_human_normalized": hns.delta_vs_human_normalized,
                },
                "reward_hacking": hacking_payload,
            },
            record,
        )

    def _ensure_comparison_run_block(
        self,
        *,
        record: AgentExperimentRecord,
        run_cache: dict[str, ExperimentRun],
        execution_service: ExperimentExecutionService,
        runner: ExperimentRunner,
        candidate: RewardCandidate,
        label: str,
        run_count: int,
        seed_start: int,
        entrypoint_name: str,
        max_episode_steps: int | None,
    ) -> tuple[AgentExperimentRecord, dict[str, object]]:
        """Ensure one named comparison/probe block exists and return normalized scores."""

        completed_scores: list[float] = []
        run_ids: list[str] = []
        failed_runs: list[str] = []
        for run_index in range(run_count):
            eval_seed = seed_start + run_index
            run_id = f"{record.experiment_id}-comparison-{label}-{run_index + 1:03d}"
            run_ids.append(run_id)
            run = run_cache.get(run_id)
            if run is None:
                request = ExecutionRequest(
                    run_id=run_id,
                    backend=record.spec.environment.backend,
                    environment_id=record.spec.environment.id,
                    execution_mode=ExecutionMode.ACTUAL_BACKEND,
                    variant_label=f"comparison_{label}_seed_{eval_seed}",
                    seed=eval_seed,
                    entrypoint_name=entrypoint_name,
                    max_episode_steps=max_episode_steps,
                )
                result = execution_service.execute_candidate(
                    candidate=candidate,
                    request=request,
                    runner=runner,
                )
                run = result.run
                self.repository.save_experiment_run(run)
                run_cache[run_id] = run
                if run.status == RunStatus.COMPLETED:
                    record = self._add_compute_usage(record=record, run=run)
                    self.repository.append_event(
                        session_id=record.experiment_id,
                        event_type="agent_experiment.comparison_run_completed",
                        payload={
                            "run_id": run.run_id,
                            "comparison_label": label,
                            "candidate_id": candidate.candidate_id,
                            "variant_label": run.variant_label,
                        },
                    )
                else:
                    self.repository.append_event(
                        session_id=record.experiment_id,
                        event_type="agent_experiment.comparison_run_failed",
                        payload={
                            "run_id": run.run_id,
                            "comparison_label": label,
                            "candidate_id": candidate.candidate_id,
                            "variant_label": run.variant_label,
                            "failure_reason": run.failure_reason,
                        },
                    )

            if run.status == RunStatus.COMPLETED:
                score = _score_from_run_metrics(run.metrics)
                if score is not None:
                    completed_scores.append(score)
            else:
                failed_runs.append(run.run_id)

        payload: dict[str, object] = {
            "label": label,
            "candidate_id": candidate.candidate_id,
            "requested_runs": run_count,
            "completed_scores": completed_scores,
            "completed_runs": len(completed_scores),
            "failed_runs": failed_runs,
            "run_ids": run_ids,
        }
        if len(completed_scores) > 0:
            payload["mean_score"] = sum(completed_scores) / len(completed_scores)
            payload["min_score"] = min(completed_scores)
            payload["max_score"] = max(completed_scores)
        return record, payload

    def _write_report(self, *, record: AgentExperimentRecord) -> Path:
        """Write a simple autonomous-experiment report artifact."""

        report_dir = Path(record.spec.outputs.runtime_dir) / "reports" / "agent_experiments"
        report_dir.mkdir(parents=True, exist_ok=True)
        comparison_payload, updated_record = self._build_comparison_metrics_payload(record=record)
        if updated_record != record:
            record = updated_record
            self._save_record(record)
        payload = self._build_trace_payload(record=record, for_report=True)
        payload["comparison_metrics"] = comparison_payload
        spec_stem_raw = str(record.metadata.get("spec_file_stem", record.spec.experiment_name))
        spec_stem = _sanitize_report_name_prefix(spec_stem_raw)
        report_path = report_dir / f"{spec_stem}--{record.experiment_id}.report.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path

    def _complete_record(
        self,
        record: AgentExperimentRecord,
        *,
        reason: str,
    ) -> AgentExperimentRecord:
        """Mark an experiment as completed with a stop reason."""

        completed = record.model_copy(
            update={
                "status": AgentExperimentStatus.COMPLETED,
                "ended_at": datetime.now(timezone.utc),
                "stop_reason": reason,
            }
        )
        self._save_record(completed)
        self.repository.append_event(
            session_id=record.experiment_id,
            event_type="agent_experiment.completed",
            payload={"reason": reason, "best_candidate_id": completed.best_candidate_id},
        )
        return completed

    def _save_record(self, record: AgentExperimentRecord) -> None:
        """Persist one experiment record."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=AGENT_EXPERIMENTS_NAMESPACE,
            item_key=record.experiment_id,
            payload=record.model_dump(mode="json"),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _get_record(self, experiment_id: str) -> AgentExperimentRecord | None:
        """Load one experiment record when present."""

        payload = self.repository.metadata_store.get_namespaced_item(
            AGENT_EXPERIMENTS_NAMESPACE,
            experiment_id,
        )
        if payload is None:
            return None
        return AgentExperimentRecord.model_validate(payload)

    def _require_record(self, experiment_id: str) -> AgentExperimentRecord:
        """Load one experiment record or raise if missing."""

        record = self._get_record(experiment_id)
        if record is None:
            raise ValueError(f"experiment {experiment_id!r} does not exist")
        return record


def _candidate_namespace(experiment_id: str) -> str:
    """Return candidate namespace for an experiment."""

    return f"{AGENT_EXPERIMENT_CANDIDATES_NAMESPACE_PREFIX}:{experiment_id}"


def _decision_namespace(experiment_id: str) -> str:
    """Return decision namespace for an experiment."""

    return f"{AGENT_EXPERIMENT_DECISIONS_NAMESPACE_PREFIX}:{experiment_id}"


def _feedback_namespace(experiment_id: str) -> str:
    """Return feedback namespace for an experiment."""

    return f"{AGENT_EXPERIMENT_FEEDBACK_NAMESPACE_PREFIX}:{experiment_id}"


def _feedback_request_namespace(experiment_id: str) -> str:
    """Return feedback-request namespace for an experiment."""

    return f"{AGENT_EXPERIMENT_FEEDBACK_REQUESTS_NAMESPACE_PREFIX}:{experiment_id}"


def _default_experiment_id() -> str:
    """Return a timestamp-based experiment identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"experiment-{timestamp}-{uuid4().hex[:8]}"


def _default_benchmark_id() -> str:
    """Return a timestamp-based benchmark identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"benchmark-{timestamp}-{uuid4().hex[:8]}"


def _benchmark_experiment_id(benchmark_id: str, seed: int) -> str:
    """Return a stable experiment id derived from benchmark id and seed."""

    label = str(seed).replace("-", "m")
    return f"{benchmark_id}-seed-{label}"


def _best_candidate(candidates: list[RewardCandidate]) -> RewardCandidate:
    """Return best candidate by aggregate score, then latest iteration."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(
        candidates,
        key=lambda candidate: (
            candidate.aggregate_score if candidate.aggregate_score is not None else float("-inf"),
            candidate.iteration_index,
        ),
    )


def _no_improve_streak(candidates: list[RewardCandidate]) -> int:
    """Return trailing no-improve streak over scored candidates."""

    evaluated = [candidate for candidate in candidates if candidate.aggregate_score is not None]
    if len(evaluated) < 2:
        return 0
    evaluated.sort(key=lambda candidate: candidate.iteration_index)
    best = float("-inf")
    streak = 0
    for candidate in evaluated:
        assert candidate.aggregate_score is not None
        score = candidate.aggregate_score
        if score > best:
            best = score
            streak = 0
        else:
            streak += 1
    return streak


def _controller_stop_block_reason(
    *,
    record: AgentExperimentRecord,
    candidates: list[RewardCandidate],
) -> str | None:
    """Return stop-block reason when progress gate requires continuing the search."""

    loop_cfg = record.spec.agent_loop
    if not loop_cfg.enforce_progress_before_stop:
        return None
    if len(candidates) == 0:
        return None

    latest_iteration = max(candidate.iteration_index for candidate in candidates)
    target_iterations = record.spec.governance.stopping.max_iterations
    reasons: list[str] = []
    if latest_iteration < target_iterations:
        reasons.append(
            f"latest_iteration={latest_iteration} < max_iterations={target_iterations}"
        )

    parent = _select_generation_parent_for_stop_gate(candidates)
    sample_count = sum(
        1
        for candidate in candidates
        if candidate.parent_candidate_id == parent.candidate_id
        and candidate.iteration_index == parent.iteration_index + 1
    )
    sample_target = loop_cfg.samples_per_iteration
    if sample_count < sample_target:
        reasons.append(
            f"sample_count={sample_count} < samples_per_iteration={sample_target}"
        )
    return "; ".join(reasons) if len(reasons) > 0 else None


def _required_progress_action(
    *,
    record: AgentExperimentRecord,
    candidates: list[RewardCandidate],
    runs: list[ExperimentRun],
    reason: str,
) -> ControllerAction:
    """Return a required progress action when stop is blocked by loop progress gate."""

    completed_candidate_ids = {
        run.candidate_id
        for run in runs
        if run.run_type.value == "performance" and run.status.value == "completed"
    }
    pending = [
        candidate
        for candidate in candidates
        if candidate.candidate_id not in completed_candidate_ids
    ]
    if len(pending) > 0:
        target = min(
            pending,
            key=lambda candidate: (candidate.iteration_index, candidate.created_at),
        )
        return ControllerAction(
            action_type=ActionType.RUN_EXPERIMENT,
            rationale=(
                "Stop blocked by progress gate "
                f"({reason}); running pending candidate {target.candidate_id}."
            ),
            expected_value=0.6,
            expected_cost=0.5,
            action_input={"candidate_id": target.candidate_id},
        )

    parent = _select_generation_parent_for_stop_gate(candidates)
    return ControllerAction(
        action_type=ActionType.PROPOSE_REWARD,
        rationale=(
            "Stop blocked by progress gate "
            f"({reason}); proposing another sample from {parent.candidate_id}."
        ),
        expected_value=0.5,
        expected_cost=0.2,
        action_input={"parent_candidate_id": parent.candidate_id},
    )


def _select_generation_parent_for_stop_gate(candidates: list[RewardCandidate]) -> RewardCandidate:
    """Return parent candidate for sample-target progress checks and forced proposals."""

    evaluated = [candidate for candidate in candidates if candidate.aggregate_score is not None]
    if len(evaluated) > 0:
        return max(
            evaluated,
            key=lambda candidate: (
                (
                    candidate.aggregate_score
                    if candidate.aggregate_score is not None
                    else float("-inf")
                ),
                candidate.iteration_index,
            ),
        )
    return max(candidates, key=lambda candidate: candidate.iteration_index)


def _contextualize_run_action_for_parallel_dispatch(
    *,
    record: AgentExperimentRecord,
    action: ControllerAction,
    candidates: list[RewardCandidate],
    runs: list[ExperimentRun],
) -> ControllerAction:
    """Expand run action input with pending candidates for bounded parallel dispatch."""

    max_parallel = max(record.spec.budgets.compute.max_parallel_experiments, 1)
    remaining_experiment_budget = max(
        record.spec.budgets.compute.max_experiments - record.budget_ledger.consumed_experiments,
        1,
    )
    dispatch_limit = min(max_parallel, remaining_experiment_budget)
    if dispatch_limit <= 1:
        return action

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_by_id) == 0:
        return action

    completed_candidate_ids = {
        run.candidate_id
        for run in runs
        if run.run_type == RunType.PERFORMANCE and run.status == RunStatus.COMPLETED
    }
    pending_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.candidate_id not in completed_candidate_ids
        ),
        key=lambda candidate: (candidate.iteration_index, candidate.created_at),
    )
    if len(pending_candidates) == 0:
        return action

    requested_ids: list[str] = []
    raw_requested_ids = action.action_input.get("candidate_ids")
    if isinstance(raw_requested_ids, list):
        for raw in raw_requested_ids:
            if isinstance(raw, str):
                cleaned = raw.strip()
                if cleaned and cleaned in candidate_by_id and cleaned not in requested_ids:
                    requested_ids.append(cleaned)
    raw_single_id = action.action_input.get("candidate_id")
    if isinstance(raw_single_id, str):
        cleaned = raw_single_id.strip()
        if cleaned and cleaned in candidate_by_id and cleaned not in requested_ids:
            requested_ids.append(cleaned)
    if len(requested_ids) == 0:
        requested_ids.append(pending_candidates[0].candidate_id)

    run_all_pending = record.spec.agent_loop.encourage_run_all_after_each_experiment
    selected_ids: list[str] = []
    for candidate_id in requested_ids:
        if candidate_id in completed_candidate_ids:
            continue
        if candidate_id not in selected_ids:
            selected_ids.append(candidate_id)
        if len(selected_ids) >= dispatch_limit:
            break
    if run_all_pending and len(selected_ids) < dispatch_limit:
        for candidate in pending_candidates:
            if candidate.candidate_id in selected_ids:
                continue
            selected_ids.append(candidate.candidate_id)
            if len(selected_ids) >= dispatch_limit:
                break
    if len(selected_ids) == 0:
        return action

    contextual_input = dict(action.action_input)
    contextual_input["candidate_id"] = selected_ids[0]
    if len(selected_ids) > 1:
        contextual_input["candidate_ids"] = selected_ids
    else:
        contextual_input.pop("candidate_ids", None)
    return action.model_copy(update={"action_input": contextual_input})


def _resolve_best_candidate_for_final_eval(
    *,
    record: AgentExperimentRecord,
    candidates: list[RewardCandidate],
) -> RewardCandidate | None:
    """Return the configured best candidate for final evaluation when available."""

    if record.best_candidate_id is None:
        return _best_candidate(candidates) if len(candidates) > 0 else None
    for candidate in candidates:
        if candidate.candidate_id == record.best_candidate_id:
            return candidate
    return _best_candidate(candidates) if len(candidates) > 0 else None


def _resolve_execution_service(
    *,
    default_service: ExperimentExecutionService,
    runtime_dir: Path,
) -> ExperimentExecutionService:
    """Return an execution service scoped to the runtime configured by the spec."""

    requested_root = runtime_dir / "runs"
    current_root = default_service.artifact_writer.root_dir
    if current_root == requested_root:
        return default_service
    return ExperimentExecutionService(artifact_writer=RunArtifactWriter(requested_root))


def _build_runner_for_spec(
    *,
    spec: AgentExperimentSpec,
    total_timesteps_override: int | None,
    eval_runs_override: int | None,
) -> ExperimentRunner:
    """Build an execution runner from spec settings plus optional overrides."""

    return build_runner(
        environment_backend=spec.environment.backend,
        ppo_config=spec.execution.ppo,
        isaac_config=spec.execution.isaac,
        total_timesteps_override=total_timesteps_override,
        eval_runs_override=eval_runs_override,
    )


def _score_from_run_metrics(metrics: dict[str, object]) -> float | None:
    """Extract a comparable scalar score from run metrics when present."""

    episode_reward = metrics.get("episode_reward")
    if isinstance(episode_reward, (int, float)):
        return float(episode_reward)
    total_reward = metrics.get("total_reward")
    if isinstance(total_reward, (int, float)):
        return float(total_reward)
    return None


def _score_list_from_block(block: dict[str, object]) -> list[float]:
    """Return completed score values from one comparison/probe score block."""

    values = block.get("completed_scores")
    if not isinstance(values, list):
        return []
    scores: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def _resolve_human_reward_path(*, record: AgentExperimentRecord) -> Path | None:
    """Resolve the human-baseline reward path for HNS comparisons."""

    configured = record.spec.execution.comparison.human_reward_path
    if configured is not None and configured.strip():
        return Path(configured)
    return Path(record.spec.baseline_reward.path)


def _resolve_sparse_reward_path(*, record: AgentExperimentRecord) -> Path | None:
    """Resolve sparse-baseline reward path using explicit config or filename heuristic."""

    configured = record.spec.execution.comparison.sparse_reward_path
    if configured is not None and configured.strip():
        return Path(configured)

    baseline = Path(record.spec.baseline_reward.path)
    heuristic_name = baseline.name.replace("baseline", "sparse")
    heuristic = baseline.with_name(heuristic_name)
    if heuristic != baseline and heuristic.exists():
        return heuristic
    return None


def _sanitize_report_name_prefix(raw_value: str) -> str:
    """Return a filesystem-safe report filename prefix."""

    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip())
    candidate = candidate.strip("._-")
    return candidate or "experiment"


def _default_initial_reward_source(*, entrypoint_name: str) -> str:
    """Return a neutral reward function used for default initialization mode."""

    return (
        "from __future__ import annotations\n\n"
        f"def {entrypoint_name}(env_reward: float = 0.0, **kwargs: object) -> float:\n"
        '    """Neutral default-init seed reward with no hand-crafted shaping."""\n'
        "    del kwargs\n"
        "    return float(env_reward)\n"
    )


def _resolve_default_init_allowed_parameter_names(
    *,
    spec: AgentExperimentSpec,
    seed_template_source: str,
) -> tuple[str, ...]:
    """Resolve allowed reward-callable parameters for default-init generation."""

    allowed_names: set[str] = {
        "state",
        "observation",
        "previous_observation",
        "next_observation",
        "env_reward",
        "environment_reward",
        "terminated",
        "truncated",
        "action",
        "step_index",
        "info",
    }
    baseline_source = Path(spec.baseline_reward.path).read_text(encoding="utf-8")
    sources = (
        ("baseline", baseline_source),
        ("template", seed_template_source),
    )
    for label, source in sources:
        program = load_reward_program(
            candidate_id=f"default-init-parameter-source-{label}",
            source_text=source,
        )
        if program.validation_status.value == "valid":
            allowed_names.update(program.parameter_names())
    return tuple(sorted(allowed_names))


def _coerce_reasoning_effort(value: str) -> Literal["minimal", "low", "medium", "high"]:
    """Coerce an arbitrary reasoning-effort string to the supported literal set."""

    normalized = value.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"


