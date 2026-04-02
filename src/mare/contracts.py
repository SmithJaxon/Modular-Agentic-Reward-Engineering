from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .manifest import ExperimentManifest


@dataclass(frozen=True)
class RunArtifact:
    """A file produced by a run."""

    name: str
    path: Path


@dataclass(frozen=True)
class RunReport:
    """Canonical output structure for an experiment run."""

    manifest: ExperimentManifest
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[RunArtifact] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["manifest"] = asdict(self.manifest)
        payload["artifacts"] = [
            {"name": artifact.name, "path": str(artifact.path)}
            for artifact in self.artifacts
        ]
        return payload


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Interface that Phase 2 will implement for IsaacGym-backed execution."""

    environment_name: str

    def train(self, manifest: ExperimentManifest) -> RunReport:
        """Train a policy under the given manifest."""

    def evaluate(self, manifest: ExperimentManifest) -> RunReport:
        """Evaluate a trained policy under the given manifest."""

    def summarize(self, report: RunReport) -> str:
        """Return a compact human-readable summary."""
