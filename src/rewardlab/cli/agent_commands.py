"""
Summary: Typer agent command implementations for run, status, events, and report.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from rewardlab.agentic.service import AgenticService

agent_app = typer.Typer(no_args_is_help=True)
SPEC_FILE_OPTION = typer.Option(..., exists=True, dir_okay=False)
JSON_OPTION = typer.Option(False, "--json")
OPTIONAL_OUTPUT_PATH = typer.Option(None)


def _data_dir() -> Path:
    """
    Resolve project-local data directory for repository files.
    """
    return Path(os.getenv("REWARDLAB_DATA_DIR", ".rewardlab"))


def _service() -> AgenticService:
    """
    Construct agentic service bound to project-local persistence.
    """
    return AgenticService.from_data_dir(_data_dir())


def _emit(payload: dict[str, Any], json_output: bool) -> None:
    """
    Emit command payload in JSON or compact human-readable format.
    """
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")


@agent_app.command("run")
def run(
    spec_file: Path = SPEC_FILE_OPTION,
    run_id: str | None = typer.Option(None),
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Execute one agentic optimization run from a specification file.
    """
    payload = _service().run(spec_file, run_id=run_id)
    _emit(payload, json_output=json_output)


@agent_app.command("status")
def status(
    run_id: str = typer.Option(...),
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Return persisted run metadata and current lifecycle status.
    """
    payload = _service().status(run_id)
    _emit(payload, json_output=json_output)


@agent_app.command("events")
def events(
    run_id: str = typer.Option(...),
    limit: int | None = typer.Option(20, min=1),
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Return recent run events from the persisted decision and tool trace.
    """
    payload = _service().events(run_id, limit=limit)
    _emit(payload, json_output=json_output)


@agent_app.command("report")
def report(
    run_id: str = typer.Option(...),
    output: Path | None = OPTIONAL_OUTPUT_PATH,
    json_output: bool = JSON_OPTION,
) -> None:
    """
    Return or export the persisted run report payload.
    """
    payload = _service().report(run_id)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload = {"run_id": run_id, "report_path": str(output)}
    _emit(payload, json_output=json_output)
