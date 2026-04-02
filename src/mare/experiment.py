from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import random
import json

from .adapters import PlaceholderIsaacGymAdapter
from .contracts import RunArtifact, RunReport
from .manifest import ExperimentManifest
from .paths import ProjectPaths


@dataclass
class RunResult:
    """Minimal run output for the Phase 1 scaffold."""

    manifest: ExperimentManifest
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    artifacts: List[str] = field(default_factory=list)


class ExperimentRunner:
    """Skeleton experiment runner for future IsaacGym integration."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def create_run_dir(self, name: str) -> Path:
        run_dir = self.paths.runs / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_manifest(self, run_dir: Path, manifest: ExperimentManifest) -> Path:
        manifest_path = run_dir / "manifest.json"
        manifest.write(manifest_path)
        return manifest_path

    def save_result(self, run_dir: Path, report: RunReport) -> Path:
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result_path

    def dry_run(self, manifest: ExperimentManifest) -> RunResult:
        """Placeholder run used until IsaacGym execution is wired in."""

        random.seed(manifest.seed)
        return RunResult(
            manifest=manifest,
            metrics={
                "seed": float(manifest.seed),
                "simulated_score": random.random(),
            },
            status="dry_run",
        )

    def placeholder_report(self, manifest: ExperimentManifest, run_dir: Path) -> RunReport:
        adapter = PlaceholderIsaacGymAdapter()
        report = adapter.evaluate(manifest)
        report_artifacts = list(report.artifacts)
        report_artifacts.append(
            RunArtifact(name="manifest", path=run_dir / "manifest.json")
        )
        report_artifacts.append(
            RunArtifact(name="result", path=run_dir / "result.json")
        )
        return RunReport(
            manifest=report.manifest,
            status=report.status,
            metrics=report.metrics,
            warnings=report.warnings,
            artifacts=report_artifacts,
            notes=report.notes,
        )
