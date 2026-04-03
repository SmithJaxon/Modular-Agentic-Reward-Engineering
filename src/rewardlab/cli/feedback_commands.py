"""
Summary: Typer feedback command implementations for human and peer review workflows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import typer

from rewardlab.cli.session_commands import JSON_OPTION, REQUIRED_TEXT_OPTION, _emit, _service

feedback_app = typer.Typer(no_args_is_help=True)
OPTIONAL_SCORE_OPTION = typer.Option(None)
OPTIONAL_ARTIFACT_OPTION = typer.Option(None)


@feedback_app.command("submit-human")
def submit_human(
    session_id: str = REQUIRED_TEXT_OPTION,
    candidate_id: str = REQUIRED_TEXT_OPTION,
    comment: str = REQUIRED_TEXT_OPTION,
    score: float | None = OPTIONAL_SCORE_OPTION,
    artifact_ref: str | None = OPTIONAL_ARTIFACT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Submit human feedback for a candidate iteration.
    """
    payload = _service().submit_human_feedback(
        session_id=session_id,
        candidate_id=candidate_id,
        comment=comment,
        score=score,
        artifact_ref=artifact_ref,
    )
    _emit(payload, json_output=json_output)


@feedback_app.command("request-peer")
def request_peer(
    session_id: str = REQUIRED_TEXT_OPTION,
    candidate_id: str = REQUIRED_TEXT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Request deterministic peer feedback from isolated review context.
    """
    payload = _service().request_peer_feedback(
        session_id=session_id,
        candidate_id=candidate_id,
    )
    _emit(payload, json_output=json_output)
