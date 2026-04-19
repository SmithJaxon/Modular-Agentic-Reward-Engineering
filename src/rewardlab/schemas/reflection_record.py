"""
Summary: Pydantic schema for iteration reflection records.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ["ReflectionRecord"]


class ReflectionRecord(BaseModel):
    """Validated representation of an iteration reflection record."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reflection_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_run_ids: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    proposed_changes: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("reflection_id", "candidate_id", "summary")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        """Reject required text fields that become empty after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("source_run_ids")
    @classmethod
    def validate_source_run_ids(cls, value: list[str]) -> list[str]:
        """Ensure source run identifiers are unique and non-empty."""

        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            if not item:
                raise ValueError("source_run_ids entries must not be blank")
            if item in seen:
                raise ValueError("source_run_ids entries must be unique")
            seen.add(item)
            normalized.append(item)
        return normalized

    @field_validator("proposed_changes")
    @classmethod
    def validate_proposed_changes(cls, value: list[str]) -> list[str]:
        """Ensure proposed changes are present and non-empty."""

        cleaned = [item for item in value if item]
        if len(cleaned) != len(value):
            raise ValueError("proposed_changes entries must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_reflection_content(self) -> ReflectionRecord:
        """Ensure the reflection contains useful content for later orchestration."""

        if not self.summary:
            raise ValueError("summary must not be blank")
        if not self.proposed_changes:
            raise ValueError("proposed_changes must not be empty")
        return self

