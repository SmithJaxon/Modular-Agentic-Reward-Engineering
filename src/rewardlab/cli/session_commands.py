"""
Summary: Typer session command implementations for start, step, pause, resume, and stop.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from rewardlab.orchestrator.session_service import SessionService
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig

session_app = typer.Typer(no_args_is_help=True)
OBJECTIVE_FILE_OPTION = typer.Option(..., exists=True, dir_okay=False)
BASELINE_FILE_OPTION = typer.Option(..., exists=True, dir_okay=False)
REQUIRED_TEXT_OPTION = typer.Option(...)
JSON_OPTION = typer.Option(False, "--json")
OPTIONAL_OUTPUT_PATH = typer.Option(None)


def _data_dir() -> Path:
    """
    Resolve project-local data directory for repository files.
    """
    return Path(os.getenv("REWARDLAB_DATA_DIR", ".rewardlab"))


def _service() -> SessionService:
    """
    Construct session service bound to project-local persistence.
    """
    return SessionService(repository=SessionRepository(_data_dir()))


def _emit(payload: dict[str, Any], json_output: bool) -> None:
    """
    Emit command payload in JSON or compact human-readable format.
    """
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")


@session_app.command("start")
def start(
    objective_file: Path = OBJECTIVE_FILE_OPTION,
    baseline_reward_file: Path = BASELINE_FILE_OPTION,
    environment_id: str = REQUIRED_TEXT_OPTION,
    environment_backend: str = REQUIRED_TEXT_OPTION,
    no_improve_limit: int = typer.Option(..., min=1),
    max_iterations: int = typer.Option(..., min=1),
    feedback_gate: str = REQUIRED_TEXT_OPTION,
    session_id: str | None = typer.Option(None),
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Start a new optimization session from objective and baseline reward files.
    """
    config = SessionConfig(
        objective_text=objective_file.read_text(encoding="utf-8").strip(),
        environment_id=environment_id,
        environment_backend=EnvironmentBackend(environment_backend),
        no_improve_limit=no_improve_limit,
        max_iterations=max_iterations,
        feedback_gate=FeedbackGate(feedback_gate),
    )
    baseline_reward = baseline_reward_file.read_text(encoding="utf-8")
    created = _service().start_session(
        config=config,
        baseline_reward_definition=baseline_reward,
        session_id=session_id,
    )
    _emit(
        {
            "session_id": created["session_id"],
            "status": created["status"],
            "created_at": created["created_at"],
        },
        json_output=json_output,
    )


@session_app.command("step")
def step(
    session_id: str = REQUIRED_TEXT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Execute one evaluate-reflect-revise iteration for a running session.
    """
    payload = _service().step_session(session_id)
    _emit(payload, json_output=json_output)


@session_app.command("pause")
def pause(
    session_id: str = REQUIRED_TEXT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Pause a currently running session.
    """
    payload = _service().pause_session(session_id)
    _emit(
        {"session_id": payload["session_id"], "status": payload["status"]},
        json_output=json_output,
    )


@session_app.command("resume")
def resume(
    session_id: str = REQUIRED_TEXT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Resume a currently paused session.
    """
    payload = _service().resume_session(session_id)
    _emit(
        {"session_id": payload["session_id"], "status": payload["status"]},
        json_output=json_output,
    )


@session_app.command("stop")
def stop(
    session_id: str = REQUIRED_TEXT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Interrupt a running session and emit best-candidate report details.
    """
    payload = _service().stop_session(session_id, report_dir=_data_dir() / "reports")
    _emit(payload, json_output=json_output)


@session_app.command("report")
def report(
    session_id: str = REQUIRED_TEXT_OPTION,
    output: Path | None = OPTIONAL_OUTPUT_PATH,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Export a session report without mutating lifecycle state.
    """
    report_dir = output.parent if output is not None else (_data_dir() / "reports")
    payload = _service().report_session(session_id, report_dir=report_dir)
    if output:
        source = Path(payload["report_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        payload["report_path"] = str(output)
    _emit(payload, json_output=json_output)
