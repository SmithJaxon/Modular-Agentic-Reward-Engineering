"""
Summary: Pydantic schema for backend runtime readiness and prerequisite reporting.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rewardlab.schemas.session_config import EnvironmentBackend

__all__ = ["BackendRuntimeStatus"]


class BackendRuntimeStatus(BaseModel):
    """Structured readiness result for a backend in the current local runtime."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    backend: EnvironmentBackend
    ready: bool
    status_reason: str = Field(min_length=1)
    missing_prerequisites: list[str] = Field(default_factory=list)
    detected_version: str | None = None

    @field_validator("status_reason")
    @classmethod
    def reject_blank_status_reason(cls, value: str) -> str:
        """Require a useful explanation for the reported runtime state."""

        if not value:
            raise ValueError("status_reason must not be blank")
        return value

    @field_validator("missing_prerequisites")
    @classmethod
    def validate_missing_prerequisites(cls, value: list[str]) -> list[str]:
        """Ensure missing prerequisite entries are non-empty and unique."""

        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            if not item:
                raise ValueError("missing_prerequisites entries must not be blank")
            if item in seen:
                raise ValueError("missing_prerequisites entries must be unique")
            seen.add(item)
            normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_runtime_state(self) -> BackendRuntimeStatus:
        """Keep the readiness state and prerequisite list consistent."""

        if self.ready and self.missing_prerequisites:
            raise ValueError("ready runtime status cannot include missing_prerequisites")
        return self
