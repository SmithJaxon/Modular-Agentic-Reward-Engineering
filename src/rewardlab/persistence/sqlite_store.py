"""
Summary: SQLite storage primitives for sessions and candidate metadata.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class SQLiteStore:
    """
    Manage low-level SQLite reads/writes for orchestrator metadata.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize a SQLite store using a project-local file path.

        Args:
            db_path: Location of SQLite file.
        """
        self._db_path = db_path

    @contextmanager
    def _connection(self) -> Any:
        """
        Yield a connection with dictionary-style row access.

        Yields:
            Active sqlite3 connection.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        """
        Create required SQLite tables if they do not already exist.
        """
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    objective_text TEXT NOT NULL,
                    environment_id TEXT NOT NULL,
                    environment_backend TEXT NOT NULL,
                    no_improve_limit INTEGER NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    feedback_gate TEXT NOT NULL,
                    best_candidate_id TEXT,
                    stop_reason TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    iteration_index INTEGER NOT NULL,
                    reward_definition TEXT NOT NULL,
                    change_summary TEXT NOT NULL,
                    aggregate_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    proposed_changes_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
                )
                """
            )

    def insert_session(self, payload: dict[str, Any]) -> None:
        """
        Insert a session metadata record.

        Args:
            payload: Session payload values with required fields.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, status, objective_text, environment_id, environment_backend,
                    no_improve_limit, max_iterations, feedback_gate, best_candidate_id,
                    stop_reason, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["session_id"],
                    payload["status"],
                    payload["objective_text"],
                    payload["environment_id"],
                    payload["environment_backend"],
                    payload["no_improve_limit"],
                    payload["max_iterations"],
                    payload["feedback_gate"],
                    payload.get("best_candidate_id"),
                    payload.get("stop_reason"),
                    json.dumps(payload.get("metadata", {}), separators=(",", ":")),
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Fetch one session row by identifier.

        Args:
            session_id: Session identifier.

        Returns:
            Session row dictionary or None when not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        return result

    def update_session_fields(self, session_id: str, values: dict[str, Any]) -> None:
        """
        Update mutable fields on a session record.

        Args:
            session_id: Session identifier.
            values: Mapping of columns to update values.
        """
        if not values:
            return
        normalized = dict(values)
        if "metadata" in normalized:
            normalized["metadata_json"] = json.dumps(
                normalized.pop("metadata"),
                separators=(",", ":"),
            )
        assignments = ", ".join(f"{key} = ?" for key in normalized)
        args = list(normalized.values()) + [session_id]
        with self._connection() as conn:
            conn.execute(f"UPDATE sessions SET {assignments} WHERE session_id = ?", args)

    def insert_candidate(self, payload: dict[str, Any]) -> None:
        """
        Insert a candidate metadata record.

        Args:
            payload: Candidate payload values.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id, session_id, iteration_index, reward_definition,
                    change_summary, aggregate_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["candidate_id"],
                    payload["session_id"],
                    payload["iteration_index"],
                    payload["reward_definition"],
                    payload["change_summary"],
                    payload["aggregate_score"],
                    payload["created_at"],
                ),
            )

    def list_candidates(self, session_id: str) -> list[dict[str, Any]]:
        """
        List candidate records for a session in iteration order.

        Args:
            session_id: Session identifier.

        Returns:
            Candidate row dictionaries.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM candidates
                WHERE session_id = ?
                ORDER BY iteration_index ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """
        Fetch one candidate by identifier.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Candidate row or None when not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_candidate_fields(self, candidate_id: str, values: dict[str, Any]) -> None:
        """
        Update mutable fields on a candidate record.

        Args:
            candidate_id: Candidate identifier.
            values: Mapping of columns to update values.
        """
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        args = list(values.values()) + [candidate_id]
        with self._connection() as conn:
            conn.execute(f"UPDATE candidates SET {assignments} WHERE candidate_id = ?", args)

    def insert_reflection(self, payload: dict[str, Any]) -> None:
        """
        Insert one reflection record linked to a candidate.

        Args:
            payload: Reflection values.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO reflections (
                    reflection_id, candidate_id, summary, proposed_changes_json,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["reflection_id"],
                    payload["candidate_id"],
                    payload["summary"],
                    json.dumps(payload["proposed_changes"], separators=(",", ":")),
                    payload["confidence"],
                    payload["created_at"],
                ),
            )
