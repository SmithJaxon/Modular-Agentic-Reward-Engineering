"""
Summary: Session lifecycle service coordinating RewardLab orchestration flows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rewardlab.experiments.artifacts import RunArtifactWriter, select_primary_artifact_ref
from rewardlab.experiments.execution_service import (
    ExecutionRequest,
    ExperimentExecutionService,
    ExperimentRunner,
)
from rewardlab.experiments.gymnasium_runner import GymnasiumExperimentRunner
from rewardlab.experiments.isaacgym_runner import IsaacGymExperimentRunner
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.feedback.demo_artifacts import DemoArtifactTracker
from rewardlab.feedback.gating import FeedbackGateEvaluator
from rewardlab.feedback.human_feedback_service import HumanFeedbackService
from rewardlab.feedback.peer_feedback_client import PeerFeedbackClient
from rewardlab.llm.openai_client import OpenAIClient
from rewardlab.orchestrator.checkpointing import CheckpointManager
from rewardlab.orchestrator.iteration_engine import IterationEngine
from rewardlab.orchestrator.reporting import SessionReportWriter
from rewardlab.orchestrator.state_machine import TransitionRequest, apply_transition
from rewardlab.persistence.event_log import EventRecord
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus
from rewardlab.schemas.feedback_entry import FeedbackEntry
from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import (
    EnvironmentBackend,
    FeedbackGate,
    SessionConfig,
    SessionRecord,
    SessionStatus,
    StopReason,
)
from rewardlab.selection.policy import CandidateSelectionPolicy
from rewardlab.utils.env import load_runtime_environment

SESSION_CANDIDATES_NAMESPACE = "session_candidates"
SESSION_FEEDBACK_NAMESPACE = "session_feedback"
SESSION_REFLECTIONS_NAMESPACE = "session_reflections"


@dataclass(frozen=True)
class ServicePaths:
    """Filesystem locations used by the session service runtime."""

    data_dir: Path
    database_path: Path
    event_log_dir: Path
    checkpoint_dir: Path
    report_dir: Path

    @classmethod
    def from_environment(cls) -> ServicePaths:
        """Build runtime paths from environment variables with local defaults."""

        env = load_runtime_environment()
        data_dir = Path(env.get("REWARDLAB_DATA_DIR", ".rewardlab"))
        database_path = Path(env.get("REWARDLAB_DB_PATH", str(data_dir / "metadata.sqlite3")))
        event_log_dir = Path(env.get("REWARDLAB_EVENT_LOG_DIR", str(data_dir / "events")))
        checkpoint_dir = Path(env.get("REWARDLAB_CHECKPOINT_DIR", str(data_dir / "checkpoints")))
        report_dir = Path(env.get("REWARDLAB_REPORT_DIR", str(data_dir / "reports")))
        return cls(
            data_dir=data_dir,
            database_path=database_path,
            event_log_dir=event_log_dir,
            checkpoint_dir=checkpoint_dir,
            report_dir=report_dir,
        )

    def experiment_artifact_writer(self) -> RunArtifactWriter:
        """Build the run-artifact writer rooted under the local runtime directory."""

        return RunArtifactWriter(self.data_dir / "runs")


@dataclass(frozen=True)
class StartedSession:
    """Response payload for a successful session start."""

    session_id: str
    status: SessionStatus
    created_at: datetime

    def to_json_payload(self) -> dict[str, str]:
        """Convert the response to a JSON-ready payload."""

        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class SteppedSession:
    """Response payload for a successful session step."""

    session_id: str
    iteration_index: int
    candidate_id: str
    status: SessionStatus
    best_candidate_id: str | None

    def to_json_payload(self) -> dict[str, str | int | None]:
        """Convert the response to a JSON-ready payload."""

        return {
            "session_id": self.session_id,
            "iteration_index": self.iteration_index,
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "best_candidate_id": self.best_candidate_id,
        }


@dataclass(frozen=True)
class StoppedSession:
    """Response payload for a successful session stop or report export."""

    session_id: str
    stop_reason: StopReason
    best_candidate_id: str | None
    report_path: Path

    def to_json_payload(self) -> dict[str, str | None]:
        """Convert the response to a JSON-ready payload."""

        return {
            "session_id": self.session_id,
            "stop_reason": self.stop_reason.value,
            "best_candidate_id": self.best_candidate_id,
            "report_path": str(self.report_path),
        }


class SessionService:
    """Coordinate session lifecycle, persistence, checkpoints, and report exports."""

    def __init__(
        self,
        *,
        paths: ServicePaths,
        repository: SessionRepository | None = None,
        iteration_engine: IterationEngine | None = None,
        selection_policy: CandidateSelectionPolicy | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        report_writer: SessionReportWriter | None = None,
        experiment_execution_service: ExperimentExecutionService | None = None,
        gymnasium_runner: GymnasiumExperimentRunner | None = None,
        isaacgym_runner: IsaacGymExperimentRunner | None = None,
        robustness_runner: RobustnessRunner | None = None,
        human_feedback_service: HumanFeedbackService | None = None,
        peer_feedback_client: PeerFeedbackClient | None = None,
        feedback_gate_evaluator: FeedbackGateEvaluator | None = None,
        execution_mode: ExecutionMode = ExecutionMode.OFFLINE_TEST,
    ) -> None:
        """Construct a session service using worktree-local dependencies."""

        runtime_environment = load_runtime_environment()
        self.paths = paths
        self.repository = repository or SessionRepository(
            RepositoryPaths(
                database_path=paths.database_path,
                event_log_path=paths.event_log_dir / "events.jsonl",
            )
        )
        self.iteration_engine = iteration_engine or IterationEngine()
        self.selection_policy = selection_policy or CandidateSelectionPolicy()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(paths.checkpoint_dir)
        self.report_writer = report_writer or SessionReportWriter(paths.report_dir)
        self.experiment_execution_service = (
            experiment_execution_service
            or ExperimentExecutionService(artifact_writer=paths.experiment_artifact_writer())
        )
        self.gymnasium_runner = gymnasium_runner or GymnasiumExperimentRunner()
        self.isaacgym_runner = isaacgym_runner or IsaacGymExperimentRunner()
        self.robustness_runner = robustness_runner or _build_optional_robustness_runner(
            runtime_environment=runtime_environment,
            experiment_execution_service=self.experiment_execution_service,
            gymnasium_runner=self.gymnasium_runner,
            isaacgym_runner=self.isaacgym_runner,
        )
        self.human_feedback_service = human_feedback_service or HumanFeedbackService(
            DemoArtifactTracker(paths.report_dir / "feedback_artifacts")
        )
        self.peer_feedback_client = peer_feedback_client or PeerFeedbackClient(OpenAIClient())
        self.feedback_gate_evaluator = feedback_gate_evaluator or FeedbackGateEvaluator()
        self.execution_mode = execution_mode

    def initialize(self) -> None:
        """Initialize local persistence directories and stores."""

        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.repository.initialize()

    def start_session(
        self,
        *,
        objective_file: Path,
        baseline_reward_file: Path,
        environment_id: str,
        environment_backend: EnvironmentBackend,
        no_improve_limit: int,
        max_iterations: int,
        feedback_gate: FeedbackGate,
        session_id: str | None = None,
    ) -> StartedSession:
        """Create a running session and persist its baseline candidate."""

        objective_text = objective_file.read_text(encoding="utf-8").strip()
        baseline_reward = baseline_reward_file.read_text(encoding="utf-8").strip()
        config = SessionConfig(
            objective_text=objective_text,
            environment_id=environment_id,
            environment_backend=environment_backend,
            no_improve_limit=no_improve_limit,
            max_iterations=max_iterations,
            feedback_gate=feedback_gate,
            metadata={
                "objective_text": objective_text,
                "current_iteration": 0,
                "no_improve_streak": 0,
                "execution_mode": self.execution_mode.value,
            },
        )
        actual_session_id = session_id or _default_session_id()
        existing_session = self.repository.get_session(actual_session_id)
        if existing_session is not None:
            return self._reuse_existing_session_start(
                session=existing_session,
                config=config,
                baseline_reward=baseline_reward,
            )

        started_at = datetime.now(UTC)
        baseline_candidate = RewardCandidate(
            candidate_id=f"{actual_session_id}-candidate-000",
            session_id=actual_session_id,
            iteration_index=0,
            reward_definition=baseline_reward,
            change_summary="Baseline candidate loaded from input reward file.",
            aggregate_score=(
                self.iteration_engine.evaluate_candidate(
                    objective_text=objective_text,
                    reward_definition=baseline_reward,
                    iteration_index=0,
                )
                if self.execution_mode == ExecutionMode.OFFLINE_TEST
                else None
            ),
        )
        record = SessionRecord(
            session_id=actual_session_id,
            started_at=started_at,
            status=SessionStatus.RUNNING,
            best_candidate_id=baseline_candidate.candidate_id,
            **config.model_dump(),
        )
        self.repository.save_session(record)
        self._save_candidate(baseline_candidate)
        self.repository.append_event(
            session_id=record.session_id,
            event_type="session.started",
            payload={
                "environment_id": environment_id,
                "baseline_candidate_id": baseline_candidate.candidate_id,
            },
        )
        self.repository.append_event(
            session_id=record.session_id,
            event_type="candidate.created",
            payload={
                "candidate_id": baseline_candidate.candidate_id,
                "iteration_index": baseline_candidate.iteration_index,
            },
        )
        self._write_checkpoint(record)
        return StartedSession(
            session_id=record.session_id,
            status=record.status,
            created_at=started_at,
        )

    def step_session(self, session_id: str) -> SteppedSession:
        """Execute one iteration for a running session."""

        session = self._get_required_session(session_id)
        if session.status != SessionStatus.RUNNING:
            raise ValueError(f"session {session_id!r} is not running")

        if self.execution_mode == ExecutionMode.ACTUAL_BACKEND:
            return self._step_session_actual_backend(session)

        candidates = self.list_candidates(session_id)
        current_candidate = max(candidates, key=lambda candidate: candidate.iteration_index)
        objective_text = str(session.metadata["objective_text"])
        previous_best = self.selection_policy.select_best_candidate(
            candidates,
            assessments=_latest_assessment_map(
                self.list_robustness_assessments(session_id=session_id)
            ),
        )
        artifacts = self.iteration_engine.run_iteration(
            session_id=session_id,
            objective_text=objective_text,
            current_candidate=current_candidate,
        )
        self._save_reflection(session_id, artifacts.reflection)
        self._save_candidate(artifacts.candidate)
        self.repository.append_event(
            session_id=session_id,
            event_type="reflection.created",
            payload={
                "reflection_id": artifacts.reflection.reflection_id,
                "candidate_id": current_candidate.candidate_id,
            },
        )
        self.repository.append_event(
            session_id=session_id,
            event_type="candidate.created",
            payload={
                "candidate_id": artifacts.candidate.candidate_id,
                "iteration_index": artifacts.candidate.iteration_index,
            },
        )

        candidates = self.list_candidates(session_id)
        reflections = self.list_reflections(session_id)
        best_candidate = self.selection_policy.select_best_candidate(
            candidates,
            assessments=_latest_assessment_map(
                self.list_robustness_assessments(session_id=session_id)
            ),
        )
        no_improve_streak = int(session.metadata.get("no_improve_streak", 0))
        if best_candidate.candidate_id != previous_best.candidate_id:
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        updated_session = session.model_copy(
            update={
                "best_candidate_id": best_candidate.candidate_id,
                "metadata": {
                    **session.metadata,
                    "current_iteration": artifacts.candidate.iteration_index,
                    "no_improve_streak": no_improve_streak,
                },
            }
        )
        updated_session = self._maybe_complete_session(updated_session)
        self.repository.save_session(updated_session)
        self.repository.append_event(
            session_id=session_id,
            event_type="session.iteration_completed",
            payload={
                "candidate_id": artifacts.candidate.candidate_id,
                "iteration_index": artifacts.candidate.iteration_index,
                "run_id": artifacts.run_id,
                "best_candidate_id": best_candidate.candidate_id,
            },
        )
        self._write_checkpoint(
            updated_session,
            candidates=candidates,
            reflections=reflections,
        )
        return SteppedSession(
            session_id=session_id,
            iteration_index=artifacts.candidate.iteration_index,
            candidate_id=artifacts.candidate.candidate_id,
            status=updated_session.status,
            best_candidate_id=updated_session.best_candidate_id,
        )

    def stop_session(self, session_id: str) -> StoppedSession:
        """Interrupt a running session and export a best-candidate report."""

        session = self._get_required_session(session_id)
        if session.status == SessionStatus.RUNNING:
            session = apply_transition(
                session,
                TransitionRequest(next_status=SessionStatus.INTERRUPTED),
            )

        candidates = self.list_candidates(session_id)
        reflections = self.list_reflections(session_id)
        experiment_runs = self.list_experiment_runs(session_id=session_id)
        robustness_assessments = self.list_robustness_assessments(session_id=session_id)
        feedback_entries = self.list_feedback(session_id)
        gate_result = self.feedback_gate_evaluator.evaluate(
            feedback_gate=session.feedback_gate,
            feedback_entries=feedback_entries,
        )
        report_path = self.report_writer.write_report(
            session=session,
            candidates=candidates,
            reflections=reflections,
            feedback_entries=feedback_entries,
            gate_result=gate_result,
            experiment_runs=experiment_runs,
            robustness_assessments=robustness_assessments,
        )
        self.repository.save_session(session)
        self.repository.append_event(
            session_id=session_id,
            event_type="session.stopped",
            payload={
                "report_path": str(report_path),
                "best_candidate_id": session.best_candidate_id,
            },
        )
        self._write_checkpoint(
            session,
            candidates=candidates,
            reflections=reflections,
        )
        return StoppedSession(
            session_id=session_id,
            stop_reason=session.stop_reason or StopReason.ERROR,
            best_candidate_id=session.best_candidate_id,
            report_path=report_path,
        )

    def pause_session(self, session_id: str) -> SessionRecord:
        """Pause a running session and persist the updated status."""

        session = self._get_required_session(session_id)
        paused = apply_transition(
            session,
            TransitionRequest(next_status=SessionStatus.PAUSED),
        )
        self.repository.save_session(paused)
        return paused

    def resume_session(self, session_id: str) -> SessionRecord:
        """Resume a paused session and persist the updated status."""

        session = self._get_required_session(session_id)
        resumed = apply_transition(
            session,
            TransitionRequest(next_status=SessionStatus.RUNNING),
        )
        self.repository.save_session(resumed)
        return resumed

    def report_session(self, session_id: str) -> StoppedSession:
        """Export a report for an existing session state."""

        session = self._get_required_session(session_id)
        candidates = self.list_candidates(session_id)
        reflections = self.list_reflections(session_id)
        experiment_runs = self.list_experiment_runs(session_id=session_id)
        robustness_assessments = self.list_robustness_assessments(session_id=session_id)
        feedback_entries = self.list_feedback(session_id)
        gate_result = self.feedback_gate_evaluator.evaluate(
            feedback_gate=session.feedback_gate,
            feedback_entries=feedback_entries,
        )
        report_path = self.report_writer.write_report(
            session=session,
            candidates=candidates,
            reflections=reflections,
            feedback_entries=feedback_entries,
            gate_result=gate_result,
            experiment_runs=experiment_runs,
            robustness_assessments=robustness_assessments,
        )
        return StoppedSession(
            session_id=session_id,
            stop_reason=session.stop_reason or StopReason.ERROR,
            best_candidate_id=session.best_candidate_id,
            report_path=report_path,
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return a session record when present."""

        return self.repository.get_session(session_id)

    def list_candidates(self, session_id: str) -> list[RewardCandidate]:
        """Return stored candidates for a session ordered by iteration index."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _candidate_namespace(session_id)
        )
        candidates = [RewardCandidate.model_validate(payload) for payload in payloads]
        return sorted(candidates, key=lambda candidate: candidate.iteration_index)

    def list_reflections(self, session_id: str) -> list[ReflectionRecord]:
        """Return stored reflections for a session ordered by creation time."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _reflection_namespace(session_id)
        )
        reflections = [ReflectionRecord.model_validate(payload) for payload in payloads]
        return sorted(reflections, key=lambda reflection: reflection.created_at)

    def list_experiment_runs(
        self,
        *,
        session_id: str | None = None,
        candidate_id: str | None = None,
    ) -> list[ExperimentRun]:
        """Return persisted experiment runs for the requested session or candidate."""

        runs = self.repository.list_experiment_runs(candidate_id=candidate_id)
        if session_id is None:
            return runs
        session_prefix = f"{session_id}-"
        candidate_ids = {candidate.candidate_id for candidate in self.list_candidates(session_id)}
        return [
            run
            for run in runs
            if run.candidate_id in candidate_ids or run.run_id.startswith(session_prefix)
        ]

    def list_robustness_assessments(
        self,
        *,
        session_id: str | None = None,
        candidate_id: str | None = None,
    ) -> list[RobustnessAssessment]:
        """Return stored robustness assessments for the requested session or candidate."""

        assessments = self.repository.list_robustness_assessments(candidate_id=candidate_id)
        if session_id is None:
            return assessments
        candidate_ids = {candidate.candidate_id for candidate in self.list_candidates(session_id)}
        return [
            assessment
            for assessment in assessments
            if assessment.candidate_id in candidate_ids
        ]

    def submit_human_feedback(
        self,
        *,
        session_id: str,
        candidate_id: str,
        comment: str,
        score: float | None = None,
        artifact_ref: str | None = None,
    ) -> FeedbackEntry:
        """Create, persist, and emit a human feedback entry."""

        self._get_required_candidate(session_id, candidate_id)
        resolved_artifact_ref = artifact_ref or self._default_feedback_artifact_ref(
            session_id=session_id,
            candidate_id=candidate_id,
        )
        feedback = self.human_feedback_service.submit_feedback(
            session_id=session_id,
            candidate_id=candidate_id,
            comment=comment,
            score=score,
            artifact_ref=resolved_artifact_ref,
        )
        self._save_feedback(session_id, feedback)
        self.repository.append_event(
            session_id=session_id,
            event_type="feedback.human_submitted",
            payload={"feedback_id": feedback.feedback_id, "candidate_id": candidate_id},
        )
        return feedback

    def request_peer_feedback(
        self,
        *,
        session_id: str,
        candidate_id: str,
    ) -> FeedbackEntry:
        """Request, persist, and emit peer feedback for a candidate."""

        session = self._get_required_session(session_id)
        candidate = self._get_required_candidate(session_id, candidate_id)
        resolved_artifact_ref = self._default_feedback_artifact_ref(
            session_id=session_id,
            candidate_id=candidate_id,
        )
        feedback = self.peer_feedback_client.request_feedback(
            session_id=session_id,
            candidate_id=candidate_id,
            objective_text=str(session.metadata["objective_text"]),
            reward_definition=candidate.reward_definition,
            aggregate_score=candidate.aggregate_score,
            artifact_ref=resolved_artifact_ref,
        )
        self._save_feedback(session_id, feedback)
        self.repository.append_event(
            session_id=session_id,
            event_type="feedback.peer_requested",
            payload={"feedback_id": feedback.feedback_id, "candidate_id": candidate_id},
        )
        return feedback

    def list_feedback(self, session_id: str) -> list[FeedbackEntry]:
        """Return feedback entries for a session ordered by creation time."""

        payloads = self.repository.metadata_store.list_namespaced_items(
            _feedback_namespace(session_id)
        )
        feedback_entries = [FeedbackEntry.model_validate(payload) for payload in payloads]
        return sorted(feedback_entries, key=lambda feedback: feedback.created_at)

    def read_events(self, session_id: str) -> list[EventRecord]:
        """Return persisted event log records for a session."""

        return self.repository.read_events(session_id)

    def _step_session_actual_backend(self, session: SessionRecord) -> SteppedSession:
        """Execute one real-backend iteration and persist the resulting evidence."""

        candidates = self.list_candidates(session.session_id)
        current_candidate = max(candidates, key=lambda candidate: candidate.iteration_index)
        previous_best = self.selection_policy.select_best_candidate(
            candidates,
            assessments=_latest_assessment_map(
                self.list_robustness_assessments(session_id=session.session_id)
            ),
        )
        objective_text = str(session.metadata["objective_text"])
        planned_iteration = self.iteration_engine.plan_iteration(
            session_id=session.session_id,
            objective_text=objective_text,
            current_candidate=current_candidate,
        )
        execution_result = self.experiment_execution_service.execute_candidate(
            candidate=planned_iteration.candidate,
            request=ExecutionRequest(
                run_id=planned_iteration.run_id,
                backend=session.environment_backend,
                environment_id=session.environment_id,
                execution_mode=self.execution_mode,
                seed=_optional_int(session.metadata.get("seed")),
            ),
            runner=self._runner_for_backend(session.environment_backend),
        )
        self.repository.save_experiment_run(execution_result.run)
        self.repository.append_event(
            session_id=session.session_id,
            event_type=(
                "experiment_run.completed"
                if execution_result.run.status == RunStatus.COMPLETED
                else "experiment_run.failed"
            ),
            payload={
                "run_id": execution_result.run.run_id,
                "candidate_id": execution_result.run.candidate_id,
                "status": execution_result.run.status.value,
                "artifact_refs": execution_result.run.artifact_refs,
                "failure_reason": execution_result.run.failure_reason,
            },
        )

        if execution_result.run.status != RunStatus.COMPLETED:
            return self._pause_after_failed_execution(
                session=session,
                previous_best=previous_best,
                failure_run=execution_result.run,
            )

        executed_candidate = planned_iteration.candidate.model_copy(
            update={"aggregate_score": _score_from_run(execution_result.run)}
        )
        reflection = self.iteration_engine.build_execution_reflection(
            session_id=session.session_id,
            candidate=executed_candidate,
            run_id=execution_result.run.run_id,
            metrics=execution_result.run.metrics,
            proposed_changes=planned_iteration.proposed_changes,
        )
        self._save_candidate(executed_candidate)
        self._save_reflection(session.session_id, reflection)
        robustness_runs, assessment = self._run_robustness_if_enabled(
            session=session,
            candidate=executed_candidate,
            primary_run=execution_result.run,
        )
        self.repository.append_event(
            session_id=session.session_id,
            event_type="reflection.created",
            payload={
                "reflection_id": reflection.reflection_id,
                "candidate_id": executed_candidate.candidate_id,
            },
        )
        self.repository.append_event(
            session_id=session.session_id,
            event_type="candidate.created",
            payload={
                "candidate_id": executed_candidate.candidate_id,
                "iteration_index": executed_candidate.iteration_index,
            },
        )

        candidates = self.list_candidates(session.session_id)
        reflections = self.list_reflections(session.session_id)
        best_candidate = self.selection_policy.select_best_candidate(
            candidates,
            assessments=_latest_assessment_map(
                self.list_robustness_assessments(session_id=session.session_id)
            ),
        )
        no_improve_streak = int(session.metadata.get("no_improve_streak", 0))
        if best_candidate.candidate_id != previous_best.candidate_id:
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        updated_session = session.model_copy(
            update={
                "best_candidate_id": best_candidate.candidate_id,
                "metadata": {
                    **session.metadata,
                    "current_iteration": executed_candidate.iteration_index,
                    "no_improve_streak": no_improve_streak,
                    "last_run_id": execution_result.run.run_id,
                    "last_robustness_run_count": len(robustness_runs),
                    **(
                        {"last_robustness_assessment_id": assessment.assessment_id}
                        if assessment is not None
                        else {}
                    ),
                },
            }
        )
        updated_session = self._maybe_complete_session(updated_session)
        self.repository.save_session(updated_session)
        self.repository.append_event(
            session_id=session.session_id,
            event_type="session.iteration_completed",
            payload={
                "candidate_id": executed_candidate.candidate_id,
                "iteration_index": executed_candidate.iteration_index,
                "run_id": execution_result.run.run_id,
                "best_candidate_id": best_candidate.candidate_id,
            },
        )
        self._write_checkpoint(
            updated_session,
            candidates=candidates,
            reflections=reflections,
        )
        return SteppedSession(
            session_id=session.session_id,
            iteration_index=executed_candidate.iteration_index,
            candidate_id=executed_candidate.candidate_id,
            status=updated_session.status,
            best_candidate_id=updated_session.best_candidate_id,
        )

    def _pause_after_failed_execution(
        self,
        *,
        session: SessionRecord,
        previous_best: RewardCandidate,
        failure_run: ExperimentRun,
    ) -> SteppedSession:
        """Pause the session after an actual-backend failure and retain resumable state."""

        paused_session = apply_transition(
            session,
            TransitionRequest(
                next_status=SessionStatus.PAUSED,
                stop_reason=StopReason.API_FAILURE_PAUSE,
                best_candidate_id=previous_best.candidate_id,
            ),
        )
        paused_session = paused_session.model_copy(
            update={
                "metadata": {
                    **session.metadata,
                    "last_failed_run_id": failure_run.run_id,
                }
            }
        )
        self.repository.save_session(paused_session)
        self.repository.append_event(
            session_id=session.session_id,
            event_type="session.paused",
            payload={
                "run_id": failure_run.run_id,
                "failure_reason": failure_run.failure_reason,
            },
        )
        self._write_checkpoint(
            paused_session,
            candidates=self.list_candidates(session.session_id),
            reflections=self.list_reflections(session.session_id),
        )
        raise RuntimeError(failure_run.failure_reason or "experiment execution failed")

    def _run_robustness_if_enabled(
        self,
        *,
        session: SessionRecord,
        candidate: RewardCandidate,
        primary_run: ExperimentRun,
    ) -> tuple[list[ExperimentRun], RobustnessAssessment | None]:
        """Execute and persist robustness probes when the service is configured for them."""

        if self.robustness_runner is None:
            return [], None

        robustness_runs, assessment = self.robustness_runner.run_candidate_probes(
            candidate=candidate,
            primary_run=primary_run,
            environment_backend=session.environment_backend,
            environment_id=session.environment_id,
        )
        for run in robustness_runs:
            self.repository.save_experiment_run(run)
            self.repository.append_event(
                session_id=session.session_id,
                event_type=(
                    "experiment_run.completed"
                    if run.status == RunStatus.COMPLETED
                    else "experiment_run.failed"
                ),
                payload={
                    "run_id": run.run_id,
                    "candidate_id": run.candidate_id,
                    "run_type": run.run_type.value,
                    "variant_label": run.variant_label,
                    "status": run.status.value,
                    "artifact_refs": run.artifact_refs,
                    "failure_reason": run.failure_reason,
                },
            )
        if assessment is None:
            return robustness_runs, None

        self.repository.save_robustness_assessment(assessment)
        self.repository.append_event(
            session_id=session.session_id,
            event_type="robustness.assessed",
            payload={
                "assessment_id": assessment.assessment_id,
                "candidate_id": assessment.candidate_id,
                "primary_run_id": assessment.primary_run_id,
                "probe_run_ids": assessment.probe_run_ids,
                "risk_level": assessment.risk_level.value,
            },
        )
        return robustness_runs, assessment

    def _runner_for_backend(
        self,
        environment_backend: EnvironmentBackend,
    ) -> ExperimentRunner:
        """Return the real-backend runner configured for the requested backend."""

        if environment_backend == EnvironmentBackend.GYMNASIUM:
            return self.gymnasium_runner
        if environment_backend == EnvironmentBackend.ISAACGYM:
            return self.isaacgym_runner
        raise RuntimeError(f"unsupported environment backend: {environment_backend.value!r}")

    def _get_required_session(self, session_id: str) -> SessionRecord:
        """Load a session record and raise when it does not exist."""

        session = self.repository.get_session(session_id)
        if session is None:
            raise ValueError(f"session {session_id!r} does not exist")
        return session

    def _get_required_candidate(
        self,
        session_id: str,
        candidate_id: str,
    ) -> RewardCandidate:
        """Load a stored candidate and raise when it does not exist."""

        for candidate in self.list_candidates(session_id):
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ValueError(f"candidate {candidate_id!r} does not exist")

    def _save_candidate(self, candidate: RewardCandidate) -> None:
        """Persist a reward candidate under the session candidate namespace."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_candidate_namespace(candidate.session_id),
            item_key=candidate.candidate_id,
            payload=candidate.model_dump(mode="json"),
            updated_at=candidate.created_at.isoformat(),
        )

    def _save_reflection(self, session_id: str, reflection: ReflectionRecord) -> None:
        """Persist a reflection record under the explicit session reflection namespace."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_reflection_namespace(session_id),
            item_key=reflection.reflection_id,
            payload=reflection.model_dump(mode="json"),
            updated_at=reflection.created_at.isoformat(),
        )

    def _reuse_existing_session_start(
        self,
        *,
        session: SessionRecord,
        config: SessionConfig,
        baseline_reward: str,
    ) -> StartedSession:
        """Return the existing session when an idempotent start request is retried."""

        self._validate_idempotent_start_request(
            session=session,
            config=config,
            baseline_reward=baseline_reward,
        )
        if session.started_at is None:
            raise ValueError(f"session {session.session_id!r} exists without started_at")
        return StartedSession(
            session_id=session.session_id,
            status=session.status,
            created_at=session.started_at,
        )

    def _validate_idempotent_start_request(
        self,
        *,
        session: SessionRecord,
        config: SessionConfig,
        baseline_reward: str,
    ) -> None:
        """Ensure a repeated explicit session start matches the original session inputs."""

        comparable_fields = (
            session.objective_text == config.objective_text
            and session.environment_id == config.environment_id
            and session.environment_backend == config.environment_backend
            and session.no_improve_limit == config.no_improve_limit
            and session.max_iterations == config.max_iterations
            and session.feedback_gate == config.feedback_gate
        )
        if not comparable_fields:
            raise ValueError(
                f"session {session.session_id!r} already exists with different configuration"
            )

        baseline_candidate = next(
            (
                candidate
                for candidate in self.list_candidates(session.session_id)
                if candidate.iteration_index == 0
            ),
            None,
        )
        if baseline_candidate is None:
            raise ValueError(
                f"session {session.session_id!r} exists without a baseline candidate"
            )
        if baseline_candidate.reward_definition != baseline_reward:
            raise ValueError(
                f"session {session.session_id!r} already exists with different baseline reward"
            )

    def _save_feedback(self, session_id: str, feedback: FeedbackEntry) -> None:
        """Persist a feedback entry under the session feedback namespace."""

        self.repository.metadata_store.upsert_namespaced_item(
            namespace=_feedback_namespace(session_id),
            item_key=feedback.feedback_id,
            payload=feedback.model_dump(mode="json"),
            updated_at=feedback.created_at.isoformat(),
        )

    def _default_feedback_artifact_ref(
        self,
        *,
        session_id: str,
        candidate_id: str,
    ) -> str | None:
        """Return the preferred persisted artifact reference for candidate review flows."""

        completed_runs = [
            run
            for run in self.list_experiment_runs(
                session_id=session_id,
                candidate_id=candidate_id,
            )
            if run.status == RunStatus.COMPLETED and run.artifact_refs
        ]
        if not completed_runs:
            return None
        return select_primary_artifact_ref(completed_runs[-1].artifact_refs)

    def _write_checkpoint(
        self,
        session: SessionRecord,
        *,
        candidates: list[RewardCandidate] | None = None,
        reflections: list[ReflectionRecord] | None = None,
    ) -> Path:
        """Write a checkpoint for the latest known session state."""

        resolved_candidates = candidates or self.list_candidates(session.session_id)
        resolved_reflections = reflections or self.list_reflections(session.session_id)
        return self.checkpoint_manager.write_checkpoint(
            session=session,
            candidates=resolved_candidates,
            reflections=resolved_reflections,
        )

    def _maybe_complete_session(self, session: SessionRecord) -> SessionRecord:
        """Apply automatic completion rules after an iteration updates session metadata."""

        current_iteration = int(session.metadata.get("current_iteration", 0))
        no_improve_streak = int(session.metadata.get("no_improve_streak", 0))
        if current_iteration >= session.max_iterations:
            return apply_transition(
                session,
                TransitionRequest(
                    next_status=SessionStatus.COMPLETED,
                    stop_reason=StopReason.ITERATION_CAP,
                ),
            )
        if no_improve_streak >= session.no_improve_limit:
            return apply_transition(
                session,
                TransitionRequest(
                    next_status=SessionStatus.COMPLETED,
                    stop_reason=StopReason.CONVERGENCE,
                ),
            )
        return session


def _candidate_namespace(session_id: str) -> str:
    """Return the metadata namespace used for session candidates."""

    return f"{SESSION_CANDIDATES_NAMESPACE}:{session_id}"


def _reflection_namespace(session_id: str) -> str:
    """Return the metadata namespace used for session reflections."""

    return f"{SESSION_REFLECTIONS_NAMESPACE}:{session_id}"


def _feedback_namespace(session_id: str) -> str:
    """Return the metadata namespace used for session feedback entries."""

    return f"{SESSION_FEEDBACK_NAMESPACE}:{session_id}"


def _build_optional_robustness_runner(
    *,
    runtime_environment: dict[str, str],
    experiment_execution_service: ExperimentExecutionService,
    gymnasium_runner: GymnasiumExperimentRunner,
    isaacgym_runner: IsaacGymExperimentRunner,
) -> RobustnessRunner | None:
    """Return an env-enabled robustness runner, or `None` when disabled."""

    if not _env_truthy(runtime_environment.get("REWARDLAB_ENABLE_ROBUSTNESS")):
        return None
    probe_matrix_path = Path(
        runtime_environment.get(
            "REWARDLAB_PROBE_MATRIX_PATH",
            "tools/reward_hack_probes/probe_matrix.yaml",
        )
    )
    return RobustnessRunner(
        probe_matrix_path=probe_matrix_path,
        experiment_execution_service=experiment_execution_service,
        gymnasium_runner=gymnasium_runner,
        isaacgym_runner=isaacgym_runner,
    )


def _latest_assessment_map(
    assessments: list[RobustnessAssessment],
) -> dict[str, RobustnessAssessment]:
    """Return the latest stored robustness assessment for each candidate."""

    latest: dict[str, RobustnessAssessment] = {}
    for assessment in assessments:
        latest[assessment.candidate_id] = assessment
    return latest


def _default_session_id() -> str:
    """Return a UTC timestamp-based session identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return f"session-{timestamp}-{uuid4().hex[:8]}"


def resolve_execution_mode_from_environment() -> ExecutionMode:
    """Return the configured session execution mode from the local runtime environment."""

    env = load_runtime_environment()
    raw_value = env.get("REWARDLAB_EXECUTION_MODE", ExecutionMode.OFFLINE_TEST.value)
    try:
        return ExecutionMode(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"unsupported REWARDLAB_EXECUTION_MODE value: {raw_value!r}"
        ) from exc


def _score_from_run(run: ExperimentRun) -> float:
    """Return the candidate score derived from a completed experiment run."""

    if "episode_reward" in run.metrics:
        return float(run.metrics["episode_reward"])
    if "total_reward" in run.metrics:
        return float(run.metrics["total_reward"])
    return 0.0


def _optional_int(value: object) -> int | None:
    """Return an integer value when the metadata entry can be safely coerced."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _env_truthy(value: str | None) -> bool:
    """Return whether an environment toggle string should be treated as enabled."""

    if value is None:
        return False
    return value.strip().casefold() in {"1", "true", "yes", "on"}
