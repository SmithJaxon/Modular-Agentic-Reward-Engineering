"""
Summary: Append-only JSONL event log utilities for RewardLab sessions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _ensure_json_value(value: Any) -> JsonValue:
    """Recursively validate and normalize a value for JSON serialization."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_ensure_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_ensure_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON event fields must use string keys")
            normalized[key] = _ensure_json_value(item)
        return normalized
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _utc_timestamp() -> str:
    """Return a UTC timestamp in ISO 8601 format with second precision."""

    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Structured JSONL event entry persisted by :class:`JsonlEventLog`."""

    event_type: str
    payload: Mapping[str, Any]
    created_at: str = field(default_factory=_utc_timestamp)
    source: str | None = None
    session_id: str | None = None
    sequence: int | None = None

    def to_json_object(self) -> dict[str, JsonValue]:
        """Convert the record to a JSON-serializable object with stable keys."""

        data: dict[str, JsonValue] = {
            "created_at": self.created_at,
            "event_type": self.event_type,
            "payload": _ensure_json_value(dict(self.payload)),
        }
        if self.source is not None:
            data["source"] = self.source
        if self.session_id is not None:
            data["session_id"] = self.session_id
        if self.sequence is not None:
            data["sequence"] = self.sequence
        return data

    @classmethod
    def from_json_object(cls, data: Mapping[str, Any]) -> EventRecord:
        """Build a record from a parsed JSON object."""

        created_at = data.get("created_at")
        event_type = data.get("event_type")
        payload = data.get("payload")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("event record is missing a valid created_at value")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event record is missing a valid event_type value")
        if not isinstance(payload, dict):
            raise ValueError("event record is missing a valid payload object")

        source = data.get("source")
        session_id = data.get("session_id")
        sequence = data.get("sequence")
        if source is not None and not isinstance(source, str):
            raise ValueError("event record source must be a string when present")
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError("event record session_id must be a string when present")
        if sequence is not None and not isinstance(sequence, int):
            raise ValueError("event record sequence must be an integer when present")

        return cls(
            event_type=event_type,
            payload=payload,
            created_at=created_at,
            source=source,
            session_id=session_id,
            sequence=sequence,
        )


class JsonlEventLog:
    """Append-only JSONL event log for session and orchestration events."""

    def __init__(self, path: Path) -> None:
        """Create a log wrapper for the target JSONL file."""

        self.path = path

    def append(self, record: EventRecord) -> EventRecord:
        """Append a single record to the log and return the written record."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            record.to_json_object(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
        return record

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
        session_id: str | None = None,
        sequence: int | None = None,
    ) -> EventRecord:
        """Create and append an event record in one step."""

        record = EventRecord(
            event_type=event_type,
            payload=payload,
            source=source,
            session_id=session_id,
            sequence=sequence,
        )
        return self.append(record)

    def append_session_event(
        self,
        session_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
        sequence: int | None = None,
    ) -> EventRecord:
        """Append an event that is explicitly bound to a session identifier."""

        return self.append_event(
            event_type=event_type,
            payload=payload,
            source=source,
            session_id=session_id,
            sequence=sequence,
        )

    def iter_records(self) -> Iterator[EventRecord]:
        """Yield records from the log in append order."""

        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL event at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(data, dict):
                    raise ValueError(
                        f"invalid JSONL event at line {line_number}: expected a JSON object"
                    )
                yield EventRecord.from_json_object(data)

    def read_all(self) -> list[EventRecord]:
        """Load every record from the log into memory."""

        return list(self.iter_records())

    def read_for_session(self, session_id: str) -> list[EventRecord]:
        """Return only events associated with the provided session identifier."""

        return [
            record for record in self.iter_records() if record.session_id == session_id
        ]

def load_event_log(path: Path) -> JsonlEventLog:
    """Return a JSONL event log wrapper for the given path."""

    return JsonlEventLog(path)
