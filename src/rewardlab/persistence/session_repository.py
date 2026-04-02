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
from rewardlab.schemas.session_config import SessionRecord, session_record_from_mapping


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
