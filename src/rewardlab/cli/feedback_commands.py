"""
Summary: Typer command handlers for RewardLab feedback operations.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from rewardlab.orchestrator.session_service import (
    ServicePaths,
    SessionService,
    resolve_execution_mode_from_environment,
)

feedback_app = typer.Typer(help="Manage RewardLab human and peer feedback.")


def _build_service() -> SessionService:
    """Construct and initialize a session service from environment paths."""

    service = SessionService(
        paths=ServicePaths.from_environment(),
        execution_mode=resolve_execution_mode_from_environment(),
    )
    service.initialize()
    return service


def _emit_payload(payload: dict[str, str], as_json: bool) -> None:
    """Emit either machine-readable JSON or a compact text rendering."""

    if as_json:
        typer.echo(json.dumps(payload))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@feedback_app.command("submit-human")
def submit_human_feedback(
    session_id: Annotated[str, typer.Option(...)],
    candidate_id: Annotated[str, typer.Option(...)],
    comment: Annotated[str, typer.Option(...)],
    score: Annotated[float | None, typer.Option()] = None,
    artifact_ref: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Attach human feedback to a candidate."""

    service = _build_service()
    feedback = service.submit_human_feedback(
        session_id=session_id,
        candidate_id=candidate_id,
        comment=comment,
        score=score,
        artifact_ref=artifact_ref,
    )
    _emit_payload(
        {
            "feedback_id": feedback.feedback_id,
            "source_type": feedback.source_type.value,
            "comment": feedback.comment,
        },
        json_output,
    )


@feedback_app.command("request-peer")
def request_peer_feedback(
    session_id: Annotated[str, typer.Option(...)],
    candidate_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Request peer feedback for a candidate."""

    service = _build_service()
    feedback = service.request_peer_feedback(
        session_id=session_id,
        candidate_id=candidate_id,
    )
    _emit_payload(
        {
            "feedback_id": feedback.feedback_id,
            "source_type": feedback.source_type.value,
            "comment": feedback.comment,
        },
        json_output,
    )
