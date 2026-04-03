"""
Summary: Session report generation and export helpers for orchestrator workflows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardlab.schemas.robustness_assessment import RiskLevel
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate
from rewardlab.schemas.session_report import (
    BestCandidateReport,
    IterationReport,
    SessionReport,
    SessionStatus,
    StopReason,
)


def build_report_payload(
    session: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> SessionReport:
    """
    Build validated session report schema from session and candidate records.

    Args:
        session: Session metadata dictionary.
        candidates: Candidate metadata rows.

    Returns:
        Validated session report model.
    """
    metadata = dict(session.get("metadata", {}))
    performance_summaries = dict(metadata.get("candidate_performance_summaries", {}))
    robustness_assessments = dict(metadata.get("robustness_assessments", {}))
    feedback_counts = dict(metadata.get("candidate_feedback_counts", {}))
    feedback_summaries = dict(metadata.get("candidate_feedback_summaries", {}))
    gate = FeedbackGate(session["feedback_gate"])
    selection_summary = str(
        metadata.get("eligible_selection_summary") or metadata.get("selection_summary", "")
    ).strip()
    best_candidate_id = (
        metadata.get("eligible_best_candidate_id") or session.get("best_candidate_id")
    )
    best = next(
        (
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == best_candidate_id
        ),
        None,
    )
    if best is None and candidates:
        best = max(candidates, key=lambda candidate: candidate["aggregate_score"])
    if best is None:
        raise ValueError("cannot build session report without candidates")
    if selection_summary in {"", "No candidate selected yet."}:
        selection_summary = (
            f"Selected candidate {best['candidate_id']} via aggregate score ranking."
        )
    if gate is not FeedbackGate.NONE and not metadata.get("eligible_best_candidate_id"):
        selection_summary = (
            f"{selection_summary} Feedback gate {gate.value} not satisfied for any "
            "candidate; final recommendation pending."
        )

    iteration_items = [
        IterationReport(
            iteration_index=candidate["iteration_index"],
            candidate_id=candidate["candidate_id"],
            performance_summary=_build_iteration_summary(
                performance_summary=performance_summaries.get(
                    candidate["candidate_id"],
                    (
                        f"iteration {candidate['iteration_index']} "
                        f"score={candidate['aggregate_score']:.3f}"
                    ),
                ),
                feedback_summary=str(feedback_summaries.get(candidate["candidate_id"], "")),
            ),
            risk_level=RiskLevel(
                robustness_assessments.get(candidate["candidate_id"], {}).get(
                    "risk_level",
                    RiskLevel.LOW.value,
                )
            ),
            feedback_count=int(feedback_counts.get(candidate["candidate_id"], 0)),
        )
        for candidate in candidates
    ]
    return SessionReport(
        session_id=session["session_id"],
        status=SessionStatus(session["status"]),
        stop_reason=_resolve_stop_reason(session),
        environment_backend=EnvironmentBackend(session["environment_backend"]),
        best_candidate=BestCandidateReport(
            candidate_id=best["candidate_id"],
            aggregate_score=best["aggregate_score"],
            selection_summary=selection_summary,
            minor_robustness_risk_accepted=bool(
                metadata.get("eligible_minor_robustness_risk_accepted")
                if metadata.get("eligible_best_candidate_id")
                else metadata.get("selected_minor_robustness_risk_accepted", False)
            ),
        ),
        iterations=iteration_items,
    )


def _build_iteration_summary(
    performance_summary: str,
    feedback_summary: str,
) -> str:
    """
    Combine performance and feedback summaries for one iteration report entry.

    Args:
        performance_summary: Stored performance summary text.
        feedback_summary: Stored feedback summary text.

    Returns:
        Combined iteration summary line.
    """
    if not feedback_summary.strip():
        return performance_summary
    return f"{performance_summary} | feedback: {feedback_summary.strip()}"


def _resolve_stop_reason(session: dict[str, Any]) -> StopReason | None:
    """
    Normalize the exported stop reason for running and terminal session reports.

    Args:
        session: Session metadata dictionary.

    Returns:
        Stop reason enum for paused/terminal sessions, or None while running.
    """
    status = SessionStatus(session["status"])
    stop_reason = session.get("stop_reason")
    if stop_reason is None and status is SessionStatus.RUNNING:
        return None
    return StopReason(str(stop_reason or StopReason.ERROR.value))


def write_report(report: SessionReport, output_dir: Path) -> Path:
    """
    Write a report model to disk as formatted JSON.

    Args:
        report: Validated session report model.
        output_dir: Base output directory.

    Returns:
        Path to report artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.session_id}.report.json"
    payload = report.model_dump()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
