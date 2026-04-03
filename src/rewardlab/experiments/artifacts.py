"""
Summary: Run-scoped artifact bundle writing for real backend experiment evidence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rewardlab.schemas.experiment_run import ExecutionMode, RunStatus
from rewardlab.schemas.session_config import EnvironmentBackend

__all__ = ["RunArtifactBundle", "RunArtifactWriter", "select_primary_artifact_ref"]


@dataclass(frozen=True, slots=True)
class RunArtifactBundle:
    """Filesystem paths for the evidence emitted by one experiment run."""

    run_id: str
    root: Path
    manifest_path: Path
    metrics_path: Path
    event_trace_path: Path | None = None
    frame_dir: Path | None = None
    video_path: Path | None = None


class RunArtifactWriter:
    """Persist metrics-first artifact bundles beneath a run-scoped directory tree."""

    def __init__(self, root_dir: Path) -> None:
        """Store the root artifact directory used for experiment runs."""

        self.root_dir = root_dir

    def write_bundle(
        self,
        *,
        run_id: str,
        backend: EnvironmentBackend,
        environment_id: str,
        execution_mode: ExecutionMode,
        status: RunStatus,
        metrics: dict[str, Any],
        event_trace: list[dict[str, Any]] | None = None,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> RunArtifactBundle:
        """Write a manifest and metrics file for a completed experiment run."""

        bundle_root = self.root_dir / run_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        metrics_path = bundle_root / "metrics.json"
        manifest_path = bundle_root / "manifest.json"
        event_trace_path: Path | None = None

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if event_trace is not None:
            event_trace_path = bundle_root / "event_trace.json"
            event_trace_path.write_text(json.dumps(event_trace, indent=2), encoding="utf-8")

        manifest = {
            "run_id": run_id,
            "backend": backend.value,
            "environment_id": environment_id,
            "execution_mode": execution_mode.value,
            "status": status.value,
            "generated_at": datetime.now(UTC).isoformat(),
            "files": {
                "metrics": metrics_path.name,
                "event_trace": event_trace_path.name if event_trace_path is not None else None,
                "frame_dir": None,
                "video": None,
            },
        }
        if manifest_metadata:
            manifest["metadata"] = manifest_metadata

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return RunArtifactBundle(
            run_id=run_id,
            root=bundle_root,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            event_trace_path=event_trace_path,
        )


def select_primary_artifact_ref(artifact_refs: list[str]) -> str | None:
    """Return the preferred artifact reference for reviews and report links."""

    for suffix in ("manifest.json", "metrics.json"):
        for artifact_ref in artifact_refs:
            if artifact_ref.endswith(suffix):
                return artifact_ref
    return artifact_refs[0] if artifact_refs else None
