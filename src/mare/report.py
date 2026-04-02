from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .benchmark import BenchmarkAggregate, BenchmarkReporter
from .project import load_project_context
from .robustness import ReadinessStatus, RewardRobustnessAnalyzer


@dataclass(frozen=True)
class ProjectRunSnapshot:
    """A discovered completed run in the repository."""

    run_dir: Path
    manifest: Dict[str, Any]
    report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "manifest": self.manifest,
            "report": self.report,
        }


@dataclass(frozen=True)
class ProjectReport:
    """Final project-level report for the current repository state."""

    project_root: Path
    run_count: int
    trace_count: int
    latest_trace_path: Optional[Path]
    readiness_status: Optional[ReadinessStatus]
    benchmark_aggregate: Optional[BenchmarkAggregate]
    run_snapshots: List[ProjectRunSnapshot] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "run_count": self.run_count,
            "trace_count": self.trace_count,
            "latest_trace_path": str(self.latest_trace_path) if self.latest_trace_path is not None else None,
            "readiness_status": self.readiness_status.to_dict() if self.readiness_status is not None else None,
            "benchmark_aggregate": self.benchmark_aggregate.to_dict() if self.benchmark_aggregate is not None else None,
            "run_snapshots": [snapshot.to_dict() for snapshot in self.run_snapshots],
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class ProjectBrief:
    """Human-readable final project summary."""

    title: str
    lines: List[str]

    def to_text(self) -> str:
        return "\n".join([self.title, *[f"- {line}" for line in self.lines]]) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "lines": self.lines,
        }


class ProjectReporter:
    """Build final project-level reports from the repository artifacts."""

    def __init__(
        self,
        benchmark_reporter: Optional[BenchmarkReporter] = None,
        robustness_analyzer: Optional[RewardRobustnessAnalyzer] = None,
    ) -> None:
        self.benchmark_reporter = benchmark_reporter or BenchmarkReporter()
        self.robustness_analyzer = robustness_analyzer or RewardRobustnessAnalyzer()

    def build_report(self, project_root: Optional[Path] = None) -> ProjectReport:
        context = load_project_context(project_root)
        run_dirs = self._discover_completed_run_dirs(context.paths.runs)
        trace_paths = self._discover_traces(context.paths.runs)
        latest_trace = self._latest_path(trace_paths)
        readiness_status = self.robustness_analyzer.build_readiness_status(latest_trace) if latest_trace is not None else None
        benchmark_aggregate = self.benchmark_reporter.aggregate_runs(
            run_dirs,
            metric_priority=["evaluation_score", "training_score", "seed"],
        ) if run_dirs else None
        snapshots = [self._snapshot(run_dir) for run_dir in run_dirs]
        notes = self._build_notes(run_dirs, trace_paths, readiness_status, benchmark_aggregate)
        return ProjectReport(
            project_root=context.paths.root,
            run_count=len(run_dirs),
            trace_count=len(trace_paths),
            latest_trace_path=latest_trace,
            readiness_status=readiness_status,
            benchmark_aggregate=benchmark_aggregate,
            run_snapshots=snapshots,
            notes=notes,
        )

    def build_brief(self, project_root: Optional[Path] = None) -> ProjectBrief:
        report = self.build_report(project_root)
        lines = [
            f"Project root: {report.project_root}",
            f"Completed runs: {report.run_count}",
            f"Traces: {report.trace_count}",
            f"Latest trace: {report.latest_trace_path or 'none'}",
        ]
        if report.readiness_status is not None:
            lines.append(
                f"Phase 5 readiness: {report.readiness_status.label} ({report.readiness_status.detail})"
            )
        if report.benchmark_aggregate is not None:
            lines.append(
                f"Benchmark aggregate: {report.benchmark_aggregate.run_count} runs, environment={report.benchmark_aggregate.environment or 'mixed'}"
            )
            for metric in report.benchmark_aggregate.metric_priority:
                values = report.benchmark_aggregate.metric_summary.get(metric, {})
                lines.append(
                    "{0}: min={1}; max={2}; avg={3}".format(
                        metric,
                        self._format_optional(values.get("min")),
                        self._format_optional(values.get("max")),
                        self._format_optional(values.get("avg")),
                    )
                )
        if report.notes:
            lines.append(f"Notes: {report.notes}")
        lines.append(
            "Status: {0}".format(
                "ready for VM-bound evaluation" if report.readiness_status and report.readiness_status.ready else "still in local scaffolding"
            )
        )
        return ProjectBrief(
            title="Phase 7 Project Brief",
            lines=lines,
        )

    def _discover_completed_run_dirs(self, runs_root: Path) -> List[Path]:
        if not runs_root.exists():
            return []
        run_dirs: List[Path] = []
        for candidate in sorted(runs_root.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / "manifest.json").exists() and (candidate / "result.json").exists():
                run_dirs.append(candidate)
        return run_dirs

    def _discover_traces(self, runs_root: Path) -> List[Path]:
        if not runs_root.exists():
            return []
        traces: List[Path] = []
        for candidate in sorted(runs_root.iterdir()):
            if not candidate.is_dir():
                continue
            trace = candidate / "orchestration.json"
            if trace.exists():
                traces.append(trace)
        return traces

    def _latest_path(self, paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        return max(paths, key=lambda path: path.stat().st_mtime)

    def _snapshot(self, run_dir: Path) -> ProjectRunSnapshot:
        manifest_path = run_dir / "manifest.json"
        result_path = run_dir / "result.json"
        return ProjectRunSnapshot(
            run_dir=run_dir,
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            report=json.loads(result_path.read_text(encoding="utf-8")),
        )

    def _build_notes(
        self,
        run_dirs: List[Path],
        trace_paths: List[Path],
        readiness_status: Optional[ReadinessStatus],
        benchmark_aggregate: Optional[BenchmarkAggregate],
    ) -> Optional[str]:
        notes: List[str] = []
        if not run_dirs:
            notes.append("No completed benchmark runs discovered.")
        if not trace_paths:
            notes.append("No orchestration traces discovered.")
        if readiness_status is None:
            notes.append("No readiness status available yet.")
        if benchmark_aggregate is None:
            notes.append("No aggregate benchmark could be computed yet.")
        return " ".join(notes) if notes else None

    def _format_optional(self, value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.6f}"
