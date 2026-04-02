"""
Summary: Repository facade for session lifecycle, candidates, reflections, and events.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from rewardlab.persistence.event_log import EventLog
from rewardlab.persistence.sqlite_store import SQLiteStore
from rewardlab.schemas.session_config import SessionConfig


class SessionRepository:
    """
    Coordinate SQLite metadata and event log operations for orchestrator state.
    """

    def __init__(self, root_dir: Path) -> None:
        """
        Initialize repository using a project-local data root.

        Args:
            root_dir: Data root containing SQLite and event files.
        """
        self._root_dir = root_dir
        self._store = SQLiteStore(root_dir / "rewardlab.sqlite3")
        self._events = EventLog(root_dir)
        self._store.initialize()

    def create_session(
        self,
        config: SessionConfig,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist a new session from validated config.

        Args:
            config: Validated session configuration.
            session_id: Optional caller-supplied session ID.

        Returns:
            Session metadata dictionary.
        """
        now = datetime.now(UTC).isoformat()
        value = session_id or f"session-{uuid4().hex[:12]}"
        payload = {
            "session_id": value,
            "status": "running",
            "objective_text": config.objective_text,
            "environment_id": config.environment_id,
            "environment_backend": config.environment_backend.value,
            "no_improve_limit": config.no_improve_limit,
            "max_iterations": config.max_iterations,
            "feedback_gate": config.feedback_gate.value,
            "best_candidate_id": None,
            "stop_reason": None,
            "metadata": config.metadata,
            "created_at": now,
            "updated_at": now,
        }
        self._store.insert_session(payload)
        self._events.append("session.created", {"config": config.model_dump()}, session_id=value)
        return payload

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Fetch one session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session metadata dictionary or None.
        """
        return self._store.get_session(session_id)

    def update_session(self, session_id: str, **fields: Any) -> None:
        """
        Update mutable fields for one session.

        Args:
            session_id: Session identifier.
            fields: Column values to update.
        """
        values = {**fields, "updated_at": datetime.now(UTC).isoformat()}
        self._store.update_session_fields(session_id, values)
        self._events.append("session.updated", {"fields": fields}, session_id=session_id)

    def add_candidate(
        self,
        session_id: str,
        iteration_index: int,
        reward_definition: str,
        change_summary: str,
        aggregate_score: float,
    ) -> dict[str, Any]:
        """
        Insert a candidate and emit an event record.

        Args:
            session_id: Session identifier.
            iteration_index: Candidate iteration index.
            reward_definition: Candidate reward code or text.
            change_summary: Human-readable change summary.
            aggregate_score: Selection score value.

        Returns:
            Candidate metadata dictionary.
        """
        payload = {
            "candidate_id": f"cand-{uuid4().hex[:12]}",
            "session_id": session_id,
            "iteration_index": iteration_index,
            "reward_definition": reward_definition,
            "change_summary": change_summary,
            "aggregate_score": aggregate_score,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._store.insert_candidate(payload)
        self._events.append("candidate.added", payload, session_id=session_id)
        return payload

    def list_candidates(self, session_id: str) -> list[dict[str, Any]]:
        """
        List session candidates in iteration order.

        Args:
            session_id: Session identifier.

        Returns:
            Candidate metadata rows.
        """
        return self._store.list_candidates(session_id)

    def get_best_candidate(self, session_id: str) -> dict[str, Any] | None:
        """
        Resolve best candidate row based on session pointer.

        Args:
            session_id: Session identifier.

        Returns:
            Best candidate row or None when absent.
        """
        session = self.get_session(session_id)
        if not session or not session.get("best_candidate_id"):
            return None
        return self._store.get_candidate(session["best_candidate_id"])

    def set_best_candidate(self, session_id: str, candidate_id: str) -> None:
        """
        Assign best candidate pointer on session metadata.

        Args:
            session_id: Session identifier.
            candidate_id: Candidate identifier.
        """
        self.update_session(session_id, best_candidate_id=candidate_id)
        self._events.append(
            "candidate.best_selected",
            {"candidate_id": candidate_id},
            session_id=session_id,
        )

    def add_reflection(
        self,
        candidate_id: str,
        summary: str,
        proposed_changes: list[str],
        confidence: float,
    ) -> dict[str, Any]:
        """
        Insert reflection metadata and emit a reflection event.

        Args:
            candidate_id: Candidate identifier.
            summary: Reflection summary text.
            proposed_changes: Proposed follow-up modifications.
            confidence: Confidence score between 0.0 and 1.0.

        Returns:
            Reflection metadata dictionary.
        """
        payload = {
            "reflection_id": f"refl-{uuid4().hex[:12]}",
            "candidate_id": candidate_id,
            "summary": summary,
            "proposed_changes": proposed_changes,
            "confidence": confidence,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._store.insert_reflection(payload)
        self._events.append("reflection.added", payload)
        return payload

    def checkpoint(self, session_id: str, payload: dict[str, Any]) -> Path:
        """
        Write and event-log a checkpoint payload.

        Args:
            session_id: Session identifier.
            payload: Serializable checkpoint payload.

        Returns:
            Path to checkpoint artifact.
        """
        path = self._events.checkpoint(session_id, payload)
        self._events.append("session.checkpointed", {"path": str(path)}, session_id=session_id)
        return path

    def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """
        Load checkpoint payload for session if present.

        Args:
            session_id: Session identifier.

        Returns:
            Parsed checkpoint payload or None.
        """
        return self._events.load_checkpoint(session_id)
