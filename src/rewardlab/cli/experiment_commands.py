"""
Summary: Typer command handlers for autonomous agent experiment operations.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from rewardlab.agentic.service import AgentExperimentService

experiment_app = typer.Typer(help="Manage autonomous tool-calling experiments.")


def _build_service() -> AgentExperimentService:
    """Construct and initialize the autonomous experiment service."""

    return AgentExperimentService.from_environment()


def _emit_payload(payload: dict[str, Any], as_json: bool) -> None:
    """Emit JSON or compact key/value output."""

    if as_json:
        typer.echo(json.dumps(payload))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@experiment_app.command("validate")
def validate_experiment_spec(
    file: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate an autonomous experiment spec file."""

    service = _build_service()
    spec = service.validate_spec(spec_file=file)
    _emit_payload(
        {
            "valid": True,
            "version": spec.version,
            "experiment_name": spec.experiment_name,
            "environment_id": spec.environment.id,
            "environment_backend": spec.environment.backend.value,
        },
        json_output,
    )


@experiment_app.command("run")
def run_experiment(
    file: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    experiment_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Start and execute an autonomous experiment."""

    service = _build_service()
    result = service.run_experiment(spec_file=file, experiment_id=experiment_id)
    _emit_payload(result.to_json_payload(), json_output)


@experiment_app.command("status")
def experiment_status(
    experiment_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read status for an autonomous experiment."""

    service = _build_service()
    result = service.get_status(experiment_id=experiment_id)
    _emit_payload(result.to_json_payload(), json_output)


@experiment_app.command("trace")
def experiment_trace(
    experiment_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Export full trace payload for an autonomous experiment."""

    service = _build_service()
    payload = service.trace_payload(experiment_id=experiment_id)
    _emit_payload(payload, json_output)


@experiment_app.command("stop")
def stop_experiment(
    experiment_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Stop an autonomous experiment."""

    service = _build_service()
    payload = service.stop_experiment(experiment_id=experiment_id)
    _emit_payload(payload.to_json_payload(), json_output)


@experiment_app.command("resume")
def resume_experiment(
    experiment_id: Annotated[str, typer.Option(...)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume a paused autonomous experiment."""

    service = _build_service()
    payload = service.resume_experiment(experiment_id=experiment_id)
    _emit_payload(payload.to_json_payload(), json_output)


@experiment_app.command("submit-human-feedback")
def submit_human_feedback(
    experiment_id: Annotated[str, typer.Option(...)],
    candidate_id: Annotated[str, typer.Option(...)],
    comment: Annotated[str, typer.Option(...)],
    score: Annotated[float | None, typer.Option()] = None,
    request_id: Annotated[str | None, typer.Option()] = None,
    artifact_ref: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Submit one human feedback entry for an autonomous experiment."""

    service = _build_service()
    payload = service.submit_human_feedback(
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        comment=comment,
        score=score,
        request_id=request_id,
        artifact_ref=artifact_ref,
    )
    _emit_payload(payload.to_json_payload(), json_output)


@experiment_app.command("benchmark-run")
def benchmark_run(
    file: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    seed: Annotated[list[int] | None, typer.Option("--seed")] = None,
    benchmark_id: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run a multi-seed benchmark and export aggregate metrics."""

    service = _build_service()
    payload = service.run_benchmark(
        spec_file=file,
        seeds=seed if seed is not None and len(seed) > 0 else None,
        benchmark_id=benchmark_id,
    )
    _emit_payload(payload.to_json_payload(), json_output)
