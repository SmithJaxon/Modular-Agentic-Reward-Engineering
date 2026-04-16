"""
Summary: Persistence helpers for agentic run metadata, events, and reports.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AgenticRepository:
    """
    Persist agentic runs as run-scoped JSON and JSONL artifacts.
    """

    def __init__(self, root_dir: Path) -> None:
        """
        Initialize repository root under `<data_dir>/agentic_runs`.
        """
        self._root_dir = root_dir / "agentic_runs"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """
        Expose repository root for run-scoped artifact storage.
        """
        return self._root_dir

    def create_run(
        self,
        *,
        run_id: str,
        spec_payload: dict[str, Any],
        spec_path: str,
    ) -> dict[str, Any]:
        """
        Create an initial run metadata record and directories.
        """
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        payload = {
            "run_id": run_id,
            "status": "running",
            "stop_reason": None,
            "turn_count": 0,
            "spec_path": spec_path,
            "spec": spec_payload,
            "created_at": now,
            "updated_at": now,
            "report_path": None,
        }
        self._write_json(self._run_metadata_path(run_id), payload)
        return payload

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Load run metadata when present.
        """
        path = self._run_metadata_path(run_id)
        if not path.exists():
            return None
        payload = self._read_json(path)
        return payload if isinstance(payload, dict) else None

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        """
        Update run metadata fields and refresh update timestamp.
        """
        current = self.get_run(run_id)
        if current is None:
            raise RuntimeError(f"run not found: {run_id}")
        current.update(fields)
        current["updated_at"] = datetime.now(UTC).isoformat()
        self._write_json(self._run_metadata_path(run_id), current)
        return current

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Append one run-scoped JSONL event row.
        """
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "payload": payload,
        }
        event_path = self._run_event_path(run_id)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")

    def list_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Load run-scoped events in chronological order.
        """
        path = self._run_event_path(run_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw = json.loads(line)
                    if isinstance(raw, dict):
                        rows.append(raw)
        if limit is None:
            return rows
        return rows[-max(0, limit) :]

    def write_report(self, run_id: str, payload: dict[str, Any]) -> Path:
        """
        Write one run report JSON payload and return its path.
        """
        path = self._run_report_path(run_id)
        self._write_json(path, payload)
        return path

    def load_report(self, run_id: str) -> dict[str, Any] | None:
        """
        Load one run report payload when present.
        """
        path = self._run_report_path(run_id)
        if not path.exists():
            return None
        payload = self._read_json(path)
        return payload if isinstance(payload, dict) else None

    def _run_dir(self, run_id: str) -> Path:
        """
        Resolve concrete run directory path.
        """
        return self._root_dir / run_id

    def _run_metadata_path(self, run_id: str) -> Path:
        """
        Resolve run metadata JSON path.
        """
        return self._run_dir(run_id) / "run.json"

    def _run_event_path(self, run_id: str) -> Path:
        """
        Resolve run event stream JSONL path.
        """
        return self._run_dir(run_id) / "events.jsonl"

    def _run_report_path(self, run_id: str) -> Path:
        """
        Resolve run report JSON path.
        """
        return self._run_dir(run_id) / "report.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """
        Write one JSON payload with stable formatting.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> object:
        """
        Parse and return one JSON payload from disk.
        """
        return json.loads(path.read_text(encoding="utf-8"))
