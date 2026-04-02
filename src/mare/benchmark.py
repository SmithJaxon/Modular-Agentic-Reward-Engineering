from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .contracts import RunArtifact, RunReport
from .manifest import ExperimentManifest


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """Loaded run artifact bundle for benchmark comparison."""

    run_dir: Path
    manifest: ExperimentManifest
    report: RunReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "manifest": asdict(self.manifest),
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkMetricDelta:
    """Comparison for a single scalar metric."""

    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    better_is_higher: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkComparison:
    """Side-by-side comparison between two runs."""

    baseline: BenchmarkRunRecord
    candidate: BenchmarkRunRecord
    deltas: List[BenchmarkMetricDelta] = field(default_factory=list)
    winner: Optional[str] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
            "winner": self.winner,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Phase 6 benchmark output for a baseline-vs-candidate comparison."""

    environment: Optional[str]
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    metric_priority: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "metric_priority": self.metric_priority,
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class BenchmarkBrief:
    """Human-readable benchmark summary."""

    title: str
    lines: List[str]

    def to_text(self) -> str:
        return "\n".join([self.title, *[f"- {line}" for line in self.lines]]) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "lines": self.lines,
        }


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Aggregate view over multiple benchmark run records."""

    environment: Optional[str]
    run_count: int
    run_names: List[str]
    metric_priority: List[str]
    metric_summary: Dict[str, Dict[str, Optional[float]]]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "run_count": self.run_count,
            "run_names": self.run_names,
            "metric_priority": self.metric_priority,
            "metric_summary": self.metric_summary,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BenchmarkAggregateBrief:
    """Human-readable benchmark aggregate summary."""

    title: str
    lines: List[str]

    def to_text(self) -> str:
        return "\n".join([self.title, *[f"- {line}" for line in self.lines]]) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "lines": self.lines,
        }


class BenchmarkReporter:
    """Load run artifacts and build a small benchmark report."""

    def load_run_record(self, run_dir: Path) -> BenchmarkRunRecord:
        manifest_path = run_dir / "manifest.json"
        result_path = run_dir / "result.json"
        manifest = self._load_manifest(manifest_path)
        report = self._load_report(result_path, manifest)
        return BenchmarkRunRecord(run_dir=run_dir, manifest=manifest, report=report)

    def compare_runs(
        self,
        baseline_run_dir: Path,
        candidate_run_dir: Path,
        metric_priority: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        baseline = self.load_run_record(baseline_run_dir)
        candidate = self.load_run_record(candidate_run_dir)
        metrics = metric_priority or self._default_metric_priority(baseline.report, candidate.report)
        deltas = [self._metric_delta(metric, baseline.report, candidate.report) for metric in metrics]
        winner = self._winner(deltas, baseline, candidate)
        summary = self._build_summary(baseline, candidate, deltas, winner)
        environment = baseline.manifest.environment if baseline.manifest.environment == candidate.manifest.environment else None
        return BenchmarkReport(
            environment=environment,
            comparisons=[
                BenchmarkComparison(
                    baseline=baseline,
                    candidate=candidate,
                    deltas=deltas,
                    winner=winner,
                    summary=summary,
                )
            ],
            metric_priority=metrics,
            notes=self._build_notes(baseline, candidate),
        )

    def build_brief(
        self,
        baseline_run_dir: Path,
        candidate_run_dir: Path,
        metric_priority: Optional[List[str]] = None,
    ) -> BenchmarkBrief:
        report = self.compare_runs(baseline_run_dir, candidate_run_dir, metric_priority=metric_priority)
        comparison = report.comparisons[0] if report.comparisons else None
        lines = [
            f"Baseline run: {baseline_run_dir.name}",
            f"Candidate run: {candidate_run_dir.name}",
        ]
        if report.environment is not None:
            lines.append(f"Environment: {report.environment}")
        if comparison is not None:
            lines.append(f"Winner: {comparison.winner or 'none'}")
            for delta in comparison.deltas:
                if delta.delta is None:
                    lines.append(f"{delta.metric}: unavailable")
                else:
                    lines.append(f"{delta.metric}: {delta.delta:+.6f}")
            lines.append(f"Summary: {comparison.summary}")
        if report.notes:
            lines.append(f"Notes: {report.notes}")
        return BenchmarkBrief(
            title="Benchmark Brief",
            lines=lines,
        )

    def aggregate_runs(self, run_dirs: List[Path], metric_priority: Optional[List[str]] = None) -> BenchmarkAggregate:
        records = [self.load_run_record(run_dir) for run_dir in run_dirs]
        environment = records[0].manifest.environment if records and all(
            record.manifest.environment == records[0].manifest.environment for record in records
        ) else None
        metric_priority = metric_priority or self._aggregate_metric_priority(records)
        metric_summary = {
            metric: self._aggregate_metric_values(metric, records) for metric in metric_priority
        }
        notes = None
        if environment is None and records:
            notes = "Environment mismatch across run directories; aggregate is informational only."
        return BenchmarkAggregate(
            environment=environment,
            run_count=len(records),
            run_names=[record.run_dir.name for record in records],
            metric_priority=metric_priority,
            metric_summary=metric_summary,
            notes=notes,
        )

    def build_aggregate_brief(
        self,
        run_dirs: List[Path],
        metric_priority: Optional[List[str]] = None,
    ) -> BenchmarkAggregateBrief:
        aggregate = self.aggregate_runs(run_dirs, metric_priority=metric_priority)
        lines = [
            f"Run count: {aggregate.run_count}",
            f"Runs: {', '.join(aggregate.run_names) if aggregate.run_names else 'none'}",
        ]
        if aggregate.environment is not None:
            lines.append(f"Environment: {aggregate.environment}")
        for metric in aggregate.metric_priority:
            metric_values = aggregate.metric_summary.get(metric, {})
            lines.append(
                "{0}: min={1}; max={2}; avg={3}".format(
                    metric,
                    self._format_optional(metric_values.get("min")),
                    self._format_optional(metric_values.get("max")),
                    self._format_optional(metric_values.get("avg")),
                )
            )
        if aggregate.notes:
            lines.append(f"Notes: {aggregate.notes}")
        return BenchmarkAggregateBrief(
            title="Benchmark Aggregate Brief",
            lines=lines,
        )

    def _load_manifest(self, path: Path) -> ExperimentManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentManifest(
            name=str(payload["name"]),
            environment=str(payload["environment"]),
            baseline=str(payload["baseline"]),
            seed=int(payload["seed"]),
            created_at=str(payload.get("created_at", "")),
            notes=payload.get("notes"),
            extra=dict(payload.get("extra") or {}),
            reward_candidate=payload.get("reward_candidate"),
        )

    def _load_report(self, path: Path, manifest: ExperimentManifest) -> RunReport:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifacts = [
            RunArtifact(name=str(item["name"]), path=Path(item["path"]))
            for item in payload.get("artifacts", [])
        ]
        report_manifest_payload = payload.get("manifest") or {}
        report_manifest = ExperimentManifest(
            name=str(report_manifest_payload.get("name", manifest.name)),
            environment=str(report_manifest_payload.get("environment", manifest.environment)),
            baseline=str(report_manifest_payload.get("baseline", manifest.baseline)),
            seed=int(report_manifest_payload.get("seed", manifest.seed)),
            created_at=str(report_manifest_payload.get("created_at", manifest.created_at)),
            notes=report_manifest_payload.get("notes", manifest.notes),
            extra=dict(report_manifest_payload.get("extra") or manifest.extra),
            reward_candidate=report_manifest_payload.get("reward_candidate", manifest.reward_candidate),
        )
        return RunReport(
            manifest=report_manifest,
            status=str(payload.get("status", "unknown")),
            metrics={key: float(value) for key, value in (payload.get("metrics") or {}).items()},
            warnings=[str(item) for item in payload.get("warnings", [])],
            artifacts=artifacts,
            notes=payload.get("notes"),
            reward_candidate=payload.get("reward_candidate", report_manifest.reward_candidate),
        )

    def _default_metric_priority(self, baseline: RunReport, candidate: RunReport) -> List[str]:
        common = [key for key in candidate.metrics.keys() if key in baseline.metrics]
        ordered = []
        for key in ("evaluation_score", "training_score"):
            if key in common:
                ordered.append(key)
        for key in common:
            if key not in ordered:
                ordered.append(key)
        return ordered

    def _metric_delta(self, metric: str, baseline: RunReport, candidate: RunReport) -> BenchmarkMetricDelta:
        baseline_value = baseline.metrics.get(metric)
        candidate_value = candidate.metrics.get(metric)
        delta = None
        if baseline_value is not None and candidate_value is not None:
            delta = round(candidate_value - baseline_value, 6)
        return BenchmarkMetricDelta(
            metric=metric,
            baseline=baseline_value,
            candidate=candidate_value,
            delta=delta,
            better_is_higher=True,
        )

    def _winner(
        self,
        deltas: List[BenchmarkMetricDelta],
        baseline: BenchmarkRunRecord,
        candidate: BenchmarkRunRecord,
    ) -> Optional[str]:
        for delta in deltas:
            if delta.metric in {"evaluation_score", "training_score"} and delta.delta is not None:
                if delta.delta > 0:
                    return candidate.run_dir.name
                if delta.delta < 0:
                    return baseline.run_dir.name
        return None

    def _build_summary(
        self,
        baseline: BenchmarkRunRecord,
        candidate: BenchmarkRunRecord,
        deltas: List[BenchmarkMetricDelta],
        winner: Optional[str],
    ) -> str:
        metric_bits = []
        for delta in deltas:
            if delta.delta is None:
                metric_bits.append(f"{delta.metric}=unavailable")
            else:
                metric_bits.append(f"{delta.metric}={delta.delta:+.6f}")
        metric_text = ", ".join(metric_bits) if metric_bits else "no shared metrics"
        winner_text = winner or "no winner"
        return (
            f"baseline={baseline.run_dir.name}; candidate={candidate.run_dir.name}; "
            f"winner={winner_text}; metrics={metric_text}"
        )

    def _build_notes(self, baseline: BenchmarkRunRecord, candidate: BenchmarkRunRecord) -> Optional[str]:
        if baseline.manifest.environment != candidate.manifest.environment:
            return "Environment mismatch; comparison is informational only."
        return None

    def _aggregate_metric_priority(self, records: List[BenchmarkRunRecord]) -> List[str]:
        metric_sets = [set(record.report.metrics.keys()) for record in records]
        if not metric_sets:
            return []
        shared = set.intersection(*metric_sets)
        ordered: List[str] = []
        for key in ("evaluation_score", "training_score"):
            if key in shared:
                ordered.append(key)
        for key in sorted(shared):
            if key not in ordered:
                ordered.append(key)
        return ordered

    def _aggregate_metric_values(self, metric: str, records: List[BenchmarkRunRecord]) -> Dict[str, Optional[float]]:
        values = [record.report.metrics.get(metric) for record in records if metric in record.report.metrics]
        if not values:
            return {"min": None, "max": None, "avg": None}
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / float(len(values)), 6),
        }

    def _format_optional(self, value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.6f}"
