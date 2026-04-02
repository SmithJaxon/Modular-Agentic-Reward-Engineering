from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json

from .adapters import get_environment_adapter
from .contracts import RunArtifact, RunReport
from .execution import PPORunDispatcher, ExecutionReceipt
from .launch import LaunchTarget, PPORunContract
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
        report = self.placeholder_report(manifest, self.paths.runs / manifest.name)
        return RunResult(
            manifest=manifest,
            metrics=report.metrics,
            status=report.status,
            artifacts=[artifact.name for artifact in report.artifacts],
        )

    def placeholder_report(self, manifest: ExperimentManifest, run_dir: Path) -> RunReport:
        adapter = get_environment_adapter(manifest.environment)
        report = adapter.evaluate(manifest)
        report_artifacts = list(report.artifacts)
        report_artifacts.append(RunArtifact(name="manifest", path=run_dir / "manifest.json"))
        report_artifacts.append(RunArtifact(name="result", path=run_dir / "result.json"))
        return RunReport(
            manifest=report.manifest,
            status=report.status,
            metrics=report.metrics,
            warnings=report.warnings,
            artifacts=report_artifacts,
            notes=report.notes,
        )

    def build_ppo_run_contract(
        self,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> PPORunContract:
        adapter = get_environment_adapter(manifest.environment)
        return adapter.build_run_contract(
            manifest=manifest,
            run_dir=run_dir,
            launch_target=launch_target,
        )

    def preview_ppo_launch(
        self,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> ExecutionReceipt:
        contract = self.build_ppo_run_contract(manifest, run_dir, launch_target)
        dispatcher = PPORunDispatcher()
        return dispatcher.preview(contract)

    def write_ppo_launch_script(
        self,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> ExecutionReceipt:
        contract = self.build_ppo_run_contract(manifest, run_dir, launch_target)
        dispatcher = PPORunDispatcher()
        return dispatcher.render_script(contract, run_dir / "launch.sh")
