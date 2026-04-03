"""
Summary: Session report builder and writer for RewardLab session summaries.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.feedback.gating import FeedbackGateResult
from rewardlab.orchestrator.iteration_engine import build_iteration_summary
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.feedback_entry import FeedbackEntry
from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import SessionRecord, SessionStatus, StopReason
from rewardlab.schemas.session_report import (
    IterationSummary,
    ReportStatus,
    RiskLevel,
    SelectionCandidate,
    SessionReport,
)


class SessionReportWriter:
    """Build and write report artifacts for stopped or completed sessions."""

    def __init__(self, report_dir: Path) -> None:
        """Store the report directory and ensure it exists on demand."""

        self.report_dir = report_dir

    def build_report(
        self,
        *,
        session: SessionRecord,
        candidates: list[RewardCandidate],
        reflections: list[ReflectionRecord],
        feedback_entries: list[FeedbackEntry],
        gate_result: FeedbackGateResult,
        experiment_runs: list[ExperimentRun] | None = None,
    ) -> SessionReport:
        """Construct a validated session report from persisted session artifacts."""

        best_candidate = _find_best_candidate(session.best_candidate_id, candidates)
        reflection_by_candidate = {
            reflection.candidate_id: reflection for reflection in reflections
        }
        feedback_by_candidate = _group_feedback_by_candidate(feedback_entries)
        run_by_candidate = _latest_runs_by_candidate(experiment_runs or [])
        iterations = [
            IterationSummary(
                iteration_index=candidate.iteration_index,
                candidate_id=candidate.candidate_id,
                performance_summary=_performance_summary_for_candidate(
                    candidate=candidate,
                    reflection_by_candidate=reflection_by_candidate,
                    feedback_by_candidate=feedback_by_candidate,
                    run=run_by_candidate.get(candidate.candidate_id),
                ),
                risk_level=RiskLevel.LOW,
                feedback_count=len(feedback_by_candidate.get(candidate.candidate_id, [])),
            )
            for candidate in sorted(candidates, key=lambda item: item.iteration_index)
        ]
        return SessionReport(
            session_id=session.session_id,
            status=_report_status_from_session(session.status),
            stop_reason=session.stop_reason or StopReason.ERROR,
            environment_backend=session.environment_backend,
            best_candidate=SelectionCandidate(
                candidate_id=best_candidate.candidate_id,
                aggregate_score=best_candidate.aggregate_score or 0.0,
                selection_summary=_selection_summary(gate_result),
                minor_robustness_risk_accepted=best_candidate.minor_robustness_risk_accepted,
            ),
            iterations=iterations,
        )

    def write_report(
        self,
        *,
        session: SessionRecord,
        candidates: list[RewardCandidate],
        reflections: list[ReflectionRecord],
        feedback_entries: list[FeedbackEntry],
        gate_result: FeedbackGateResult,
        experiment_runs: list[ExperimentRun] | None = None,
    ) -> Path:
        """Write a JSON report artifact and return its path."""

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report(
            session=session,
            candidates=candidates,
            reflections=reflections,
            feedback_entries=feedback_entries,
            gate_result=gate_result,
            experiment_runs=experiment_runs,
        )
        report_path = self.report_dir / f"{session.session_id}.report.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return report_path


def _find_best_candidate(
    best_candidate_id: str | None,
    candidates: list[RewardCandidate],
) -> RewardCandidate:
    """Return the report's best candidate, preferring the stored identifier."""

    if not candidates:
        raise ValueError("at least one candidate is required to build a report")
    if best_candidate_id is None:
        return max(candidates, key=lambda candidate: candidate.aggregate_score or float("-inf"))
    for candidate in candidates:
        if candidate.candidate_id == best_candidate_id:
            return candidate
    raise ValueError(f"best candidate {best_candidate_id!r} was not found")


def _report_status_from_session(status: SessionStatus) -> ReportStatus:
    """Map session lifecycle states to report lifecycle states."""

    mapping = {
        SessionStatus.PAUSED: ReportStatus.PAUSED,
        SessionStatus.INTERRUPTED: ReportStatus.INTERRUPTED,
        SessionStatus.COMPLETED: ReportStatus.COMPLETED,
        SessionStatus.FAILED: ReportStatus.FAILED,
    }
    if status not in mapping:
        raise ValueError(f"session status {status.value!r} cannot be reported yet")
    return mapping[status]


def _group_feedback_by_candidate(
    feedback_entries: list[FeedbackEntry],
) -> dict[str, list[FeedbackEntry]]:
    """Group feedback entries by candidate identifier."""

    grouped: dict[str, list[FeedbackEntry]] = {}
    for entry in feedback_entries:
        grouped.setdefault(entry.candidate_id, []).append(entry)
    return grouped


def _latest_runs_by_candidate(
    experiment_runs: list[ExperimentRun],
) -> dict[str, ExperimentRun]:
    """Return the latest persisted experiment run for each candidate."""

    latest: dict[str, ExperimentRun] = {}
    for run in sorted(
        experiment_runs,
        key=lambda item: (item.ended_at or item.started_at, item.run_id),
    ):
        latest[run.candidate_id] = run
    return latest


def _performance_summary_for_candidate(
    *,
    candidate: RewardCandidate,
    reflection_by_candidate: dict[str, ReflectionRecord],
    feedback_by_candidate: dict[str, list[FeedbackEntry]],
    run: ExperimentRun | None,
) -> str:
    """Build an iteration summary that includes any stored run evidence."""

    reflection = reflection_by_candidate.get(candidate.candidate_id)
    if reflection is None and candidate.parent_candidate_id is not None:
        reflection = reflection_by_candidate.get(candidate.parent_candidate_id)

    summary = build_iteration_summary(
        candidate=candidate,
        reflection=reflection,
        feedback_entries=feedback_by_candidate.get(candidate.candidate_id, []),
    )
    if run is None:
        return summary

    artifact_refs = ", ".join(run.artifact_refs) if run.artifact_refs else "no artifacts recorded"
    return (
        f"{summary} Run {run.run_id} used {run.execution_mode.value} on "
        f"{run.environment_id}; artifacts: {artifact_refs}."
    )


def _selection_summary(gate_result: FeedbackGateResult) -> str:
    """Return the final selection summary message for the report."""

    if not gate_result.satisfied:
        return "Recommendation pending required feedback gate."
    if gate_result.conflict_detected:
        return (
            "Feedback gate satisfied with conflicting feedback; inspect human and peer "
            "notes before merge."
        )
    return (
        "Feedback gate satisfied. Highest-ranked candidate by deterministic MVP "
        "selection policy."
    )
