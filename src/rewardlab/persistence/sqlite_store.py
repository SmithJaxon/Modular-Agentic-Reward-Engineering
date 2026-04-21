"""
Summary: SQLite-backed metadata store for RewardLab session persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast


class SQLiteMetadataStore:
    """Persist session metadata records in a lightweight SQLite database."""

    def __init__(self, database_path: Path) -> None:
        """Store the SQLite database location and ensure its parent exists."""

        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Create required tables when they do not already exist."""

        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    environment_backend TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS key_value_store (
                    namespace TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, item_key)
                )
                """
            )

    def upsert_session_record(
        self,
        session_id: str,
        status: str,
        environment_backend: str,
        payload: dict[str, Any],
        updated_at: str,
    ) -> None:
        """Insert or replace the canonical metadata row for a session."""

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    status,
                    environment_backend,
                    payload_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    environment_backend = excluded.environment_backend,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, status, environment_backend, json.dumps(payload), updated_at),
            )

    def get_session_record(self, session_id: str) -> dict[str, Any] | None:
        """Return the decoded metadata payload for a session when present."""

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return _decode_payload(row["payload_json"])

    def list_session_records(self) -> list[dict[str, Any]]:
        """Return decoded session metadata payloads ordered by update time."""

        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM sessions
                ORDER BY updated_at DESC, session_id ASC
                """
            ).fetchall()

        return [_decode_payload(row["payload_json"]) for row in rows]

    def upsert_namespaced_item(
        self,
        namespace: str,
        item_key: str,
        payload: dict[str, Any],
        updated_at: str,
    ) -> None:
        """Persist an arbitrary namespaced metadata payload."""

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO key_value_store (namespace, item_key, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, item_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (namespace, item_key, json.dumps(payload), updated_at),
            )

    def get_namespaced_item(self, namespace: str, item_key: str) -> dict[str, Any] | None:
        """Fetch a decoded namespaced payload when present."""

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM key_value_store
                WHERE namespace = ? AND item_key = ?
                """,
                (namespace, item_key),
            ).fetchone()

        if row is None:
            return None
        return _decode_payload(row["payload_json"])

    def list_namespaced_items(self, namespace: str) -> list[dict[str, Any]]:
        """List decoded payloads for a namespace ordered by update time."""

        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM key_value_store
                WHERE namespace = ?
                ORDER BY updated_at DESC, item_key ASC
                """,
                (namespace,),
            ).fetchall()

        return [_decode_payload(row["payload_json"]) for row in rows]

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection configured for dictionary-like row access."""

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()


def _decode_payload(raw_payload: str) -> dict[str, Any]:
    """Decode a stored JSON object payload from SQLite."""

    decoded = json.loads(raw_payload)
    if not isinstance(decoded, dict):
        raise ValueError("stored payload must decode to a JSON object")
    return cast(dict, decoded)
