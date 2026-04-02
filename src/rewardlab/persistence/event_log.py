"""
Summary: Append-only JSONL event log and checkpoint persistence helpers.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventLog:
    """
    Persist orchestration events and checkpoints in project-local files.
    """

    def __init__(self, root_dir: Path) -> None:
        """
        Initialize file paths for event and checkpoint persistence.

        Args:
            root_dir: Base directory for persisted session artifacts.
        """
        self._root_dir = root_dir
        self._event_path = root_dir / "events.jsonl"
        self._checkpoint_dir = root_dir / "checkpoints"
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """
        Append one JSON event record to the event stream.

        Args:
            event_type: Stable event category string.
            payload: Event payload object.
            session_id: Optional session identifier.
        """
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "payload": payload,
        }
        with self._event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")

    def iter_events(self, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """
        Yield event records optionally filtered by session.

        Args:
            session_id: Optional session identifier filter.

        Yields:
            Parsed event row objects.
        """
        if not self._event_path.exists():
            return
        with self._event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if session_id and row.get("session_id") != session_id:
                    continue
                yield row

    def checkpoint(self, session_id: str, payload: dict[str, Any]) -> Path:
        """
        Write a resumable checkpoint snapshot for a session.

        Args:
            session_id: Session identifier.
            payload: Serializable checkpoint payload.

        Returns:
            Path to written checkpoint file.
        """
        path = self._checkpoint_dir / f"{session_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """
        Load a session checkpoint snapshot if present.

        Args:
            session_id: Session identifier.

        Returns:
            Parsed checkpoint payload or None when missing.
        """
        path = self._checkpoint_dir / f"{session_id}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid checkpoint payload in {path}")
        return raw
