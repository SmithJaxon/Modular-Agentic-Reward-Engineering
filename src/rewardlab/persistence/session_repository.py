"""
Summary: Repository facade that coordinates session metadata and event persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rewardlab.persistence.event_log import EventRecord, JsonlEventLog
from rewardlab.persistence.sqlite_store import SQLiteMetadataStore
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import SessionRecord, session_record_from_mapping

EXPERIMENT_RUNS_NAMESPACE = "experiment_runs"
ROBUSTNESS_ASSESSMENTS_NAMESPACE = "robustness_assessments"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-formatted string."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RepositoryPaths:
    """Filesystem paths used by the session repository."""

    database_path: Path
    event_log_path: Path


class SessionRepository:
    """Coordinate persisted session state across SQLite and JSONL event logs."""

    def __init__(self, paths: RepositoryPaths) -> None:
        """Create repository dependencies from worktree-local paths."""

        self.paths = paths
        self.metadata_store = SQLiteMetadataStore(paths.database_path)
        self.event_log = JsonlEventLog(paths.event_log_path)

    def initialize(self) -> None:
        """Create backing storage for session metadata and events."""

        self.metadata_store.initialize()
        self.paths.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def save_session(self, record: SessionRecord) -> SessionRecord:
        """Persist a validated session record and return it unchanged."""

        payload = record.model_dump(mode="json")
        self.metadata_store.upsert_session_record(
            session_id=record.session_id,
            status=record.status.value,
            environment_backend=record.environment_backend.value,
            payload=payload,
            updated_at=_utc_now_iso(),
        )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Load a session record by identifier when present."""

        payload = self.metadata_store.get_session_record(session_id)
        if payload is None:
            return None
        return session_record_from_mapping(payload)

    def list_sessions(self) -> list[SessionRecord]:
        """Return all persisted session records in most-recent-first order."""

        payloads = self.metadata_store.list_session_records()
        return [session_record_from_mapping(payload) for payload in payloads]

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventRecord:
        """Append an event to the JSONL log and return the persisted event."""

        return self.event_log.append_session_event(
            session_id=session_id,
            event_type=event_type,
            payload=payload,
        )

    def read_events(self, session_id: str | None = None) -> list[EventRecord]:
        """Read logged events, optionally filtered to a single session."""

        events = self.event_log.read_all()
        if session_id is None:
            return events
        return self.event_log.read_for_session(session_id)

    def save_experiment_run(self, run: ExperimentRun) -> ExperimentRun:
        """Persist a validated experiment run and return it unchanged."""

        self.metadata_store.upsert_namespaced_item(
            namespace=EXPERIMENT_RUNS_NAMESPACE,
            item_key=run.run_id,
            payload=run.model_dump(mode="json"),
            updated_at=_run_updated_at_iso(run),
        )
        return run

    def get_experiment_run(self, run_id: str) -> ExperimentRun | None:
        """Load an experiment run by identifier when present."""

        payload = self.metadata_store.get_namespaced_item(EXPERIMENT_RUNS_NAMESPACE, run_id)
        if payload is None:
            return None
        return ExperimentRun.model_validate(payload)

    def list_experiment_runs(self, *, candidate_id: str | None = None) -> list[ExperimentRun]:
        """Return persisted experiment runs ordered by start time and run id."""

        payloads = self.metadata_store.list_namespaced_items(EXPERIMENT_RUNS_NAMESPACE)
        runs = [ExperimentRun.model_validate(payload) for payload in payloads]
        if candidate_id is not None:
            runs = [run for run in runs if run.candidate_id == candidate_id]
        return sorted(
            runs,
            key=lambda run: (_sort_timestamp(run.started_at, run.ended_at), run.run_id),
        )

    def save_robustness_assessment(
        self,
        assessment: RobustnessAssessment,
    ) -> RobustnessAssessment:
        """Persist a robustness assessment and return it unchanged."""

        self.metadata_store.upsert_namespaced_item(
            namespace=ROBUSTNESS_ASSESSMENTS_NAMESPACE,
            item_key=assessment.assessment_id,
            payload=assessment.model_dump(mode="json"),
            updated_at=assessment.created_at.isoformat(),
        )
        return assessment

    def get_robustness_assessment(
        self,
        assessment_id: str,
    ) -> RobustnessAssessment | None:
        """Load a robustness assessment by identifier when present."""

        payload = self.metadata_store.get_namespaced_item(
            ROBUSTNESS_ASSESSMENTS_NAMESPACE,
            assessment_id,
        )
        if payload is None:
            return None
        return RobustnessAssessment.model_validate(payload)

    def list_robustness_assessments(
        self,
        *,
        candidate_id: str | None = None,
    ) -> list[RobustnessAssessment]:
        """Return persisted robustness assessments ordered by creation time."""

        payloads = self.metadata_store.list_namespaced_items(ROBUSTNESS_ASSESSMENTS_NAMESPACE)
        assessments = [RobustnessAssessment.model_validate(payload) for payload in payloads]
        if candidate_id is not None:
            assessments = [
                assessment
                for assessment in assessments
                if assessment.candidate_id == candidate_id
            ]
        return sorted(
            assessments,
            key=lambda assessment: (assessment.created_at, assessment.assessment_id),
        )


def _run_updated_at_iso(run: ExperimentRun) -> str:
    """Return the best available timestamp for experiment-run persistence updates."""

    if run.ended_at is not None:
        return run.ended_at.isoformat()
    if run.started_at is not None:
        return run.started_at.isoformat()
    return _utc_now_iso()


def _sort_timestamp(
    started_at: datetime | None,
    ended_at: datetime | None,
) -> datetime:
    """Resolve a stable timestamp for ordering persisted experiment runs."""

    return started_at or ended_at or datetime(1970, 1, 1, tzinfo=UTC)
