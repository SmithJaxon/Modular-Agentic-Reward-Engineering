"""
Summary: Typer command handlers for RewardLab session lifecycle operations.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate

session_app = typer.Typer(help="Manage RewardLab optimization sessions.")


def _build_service() -> SessionService:
    """Construct and initialize a session service from environment paths."""

    service = SessionService(paths=ServicePaths.from_environment())
    service.initialize()
    return service


def _emit_payload(payload: Mapping[str, object], as_json: bool) -> None:
    """Emit either machine-readable JSON or a compact text rendering."""

    if as_json:
        typer.echo(json.dumps(payload))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@session_app.command("start")
def start_session(
    objective_file: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    baseline_reward_file: Annotated[
        Path, typer.Option(..., exists=True, dir_okay=False)
    ],
    environment_id: Annotated[str, typer.Option(...)],
    environment_backend: Annotated[EnvironmentBackend, typer.Option(...)],
    no_improve_limit: Annotated[int, typer.Option(..., min=1)],
    max_iterations: Annotated[int, typer.Option(..., min=1)],
    feedback_gate: Annotated[FeedbackGate, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    session_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Start a new optimization session."""

    service = _build_service()
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id=environment_id,
        environment_backend=environment_backend,
        no_improve_limit=no_improve_limit,
        max_iterations=max_iterations,
        feedback_gate=feedback_gate,
        session_id=session_id,
    )
    _emit_payload(started.to_json_payload(), json_output)


@session_app.command("step")
def step_session(
    session_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Execute one deterministic iteration for a session."""

    service = _build_service()
    stepped = service.step_session(session_id)
    _emit_payload(stepped.to_json_payload(), json_output)


@session_app.command("pause")
def pause_session(
    session_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Pause a running session."""

    service = _build_service()
    session = service.pause_session(session_id)
    _emit_payload(
        {"session_id": session.session_id, "status": session.status.value},
        json_output,
    )


@session_app.command("resume")
def resume_session(
    session_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume a paused session."""

    service = _build_service()
    session = service.resume_session(session_id)
    _emit_payload(
        {"session_id": session.session_id, "status": session.status.value},
        json_output,
    )


@session_app.command("stop")
def stop_session(
    session_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Interrupt a session and export the current report."""

    service = _build_service()
    stopped = service.stop_session(session_id)
    _emit_payload(stopped.to_json_payload(), json_output)


@session_app.command("report")
def report_session(
    session_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Write a report for an existing session state."""

    service = _build_service()
    report = service.report_session(session_id)
    _emit_payload(report.to_json_payload(), json_output)
