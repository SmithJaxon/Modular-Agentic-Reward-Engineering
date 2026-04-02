"""
Summary: Session report generation and export helpers for orchestrator workflows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.schemas.session_report import (
    BestCandidateReport,
    IterationReport,
    RiskLevel,
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
    best_candidate_id = session.get("best_candidate_id")
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

    iteration_items = [
        IterationReport(
            iteration_index=candidate["iteration_index"],
            candidate_id=candidate["candidate_id"],
            performance_summary=(
                f"iteration {candidate['iteration_index']} "
                f"score={candidate['aggregate_score']:.3f}"
            ),
            risk_level=RiskLevel.LOW,
            feedback_count=0,
        )
        for candidate in candidates
    ]
    return SessionReport(
        session_id=session["session_id"],
        status=SessionStatus(session["status"]),
        stop_reason=StopReason(session["stop_reason"] or StopReason.ERROR.value),
        environment_backend=EnvironmentBackend(session["environment_backend"]),
        best_candidate=BestCandidateReport(
            candidate_id=best["candidate_id"],
            aggregate_score=best["aggregate_score"],
            selection_summary=(
                f"Selected candidate {best['candidate_id']} "
                "via aggregate score ranking."
            ),
            minor_robustness_risk_accepted=False,
        ),
        iterations=iteration_items,
    )


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
