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

from rewardlab.agentic.eureka_metrics import (
    compute_eureka_comparison_metrics,
    compute_reward_hacking_metrics,
    extract_primary_score_from_report,
    load_report_payload,
)
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


@experiment_app.command("eureka-metrics")
def eureka_metrics(
    method_report: Annotated[
        Path | None, typer.Option("--method-report", exists=True, dir_okay=False)
    ] = None,
    method_score: Annotated[float | None, typer.Option("--method-score")] = None,
    human_report: Annotated[
        Path | None, typer.Option("--human-report", exists=True, dir_okay=False)
    ] = None,
    human_score: Annotated[float | None, typer.Option("--human-score")] = None,
    sparse_report: Annotated[
        Path | None, typer.Option("--sparse-report", exists=True, dir_okay=False)
    ] = None,
    sparse_score: Annotated[float | None, typer.Option("--sparse-score")] = None,
    probe_report: Annotated[
        list[Path] | None, typer.Option("--probe-report", exists=True, dir_okay=False)
    ] = None,
    probe_score: Annotated[list[float] | None, typer.Option("--probe-score")] = None,
    clip_min: Annotated[float, typer.Option("--clip-min")] = 0.0,
    clip_max: Annotated[float, typer.Option("--clip-max")] = 3.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compute Eureka-style human-normalized metrics and reward-hacking risk."""

    resolved_method_score, method_source = _resolve_score_input(
        label="method",
        report_path=method_report,
        explicit_score=method_score,
    )
    resolved_human_score, human_source = _resolve_score_input(
        label="human",
        report_path=human_report,
        explicit_score=human_score,
    )
    resolved_sparse_score, sparse_source = _resolve_score_input(
        label="sparse",
        report_path=sparse_report,
        explicit_score=sparse_score,
    )

    comparison = compute_eureka_comparison_metrics(
        method_score=resolved_method_score,
        human_score=resolved_human_score,
        sparse_score=resolved_sparse_score,
        method_score_source=method_source,
        human_score_source=human_source,
        sparse_score_source=sparse_source,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    probe_scores = _resolve_probe_scores(
        probe_reports=probe_report,
        explicit_probe_scores=probe_score,
    )
    hacking_payload: dict[str, Any] | None = None
    if len(probe_scores) > 0:
        hacking = compute_reward_hacking_metrics(
            method_score=resolved_method_score,
            human_score=resolved_human_score,
            sparse_score=resolved_sparse_score,
            probe_scores=probe_scores,
        )
        hacking_payload = {
            "probe_count": hacking.probe_count,
            "probe_min_score": hacking.probe_min_score,
            "probe_mean_score": hacking.probe_mean_score,
            "probe_below_sparse_rate": hacking.probe_below_sparse_rate,
            "worst_score_degradation_ratio": hacking.worst_score_degradation_ratio,
            "mean_score_degradation_ratio": hacking.mean_score_degradation_ratio,
            "probe_min_human_normalized_score": hacking.probe_min_human_normalized_score,
            "probe_mean_human_normalized_score": hacking.probe_mean_human_normalized_score,
            "worst_human_normalized_drop": hacking.worst_human_normalized_drop,
            "mean_human_normalized_drop": hacking.mean_human_normalized_drop,
            "perils_relative_reward_function_performance": (
                hacking.perils_relative_reward_function_performance
            ),
            "perils_hacking_severity": hacking.perils_hacking_severity,
            "hacking_risk_index": hacking.hacking_risk_index,
            "hacking_risk_level": hacking.hacking_risk_level,
        }

    payload = {
        "method_score": comparison.method_score,
        "method_score_source": comparison.method_score_source,
        "human_score": comparison.human_score,
        "human_score_source": comparison.human_score_source,
        "sparse_score": comparison.sparse_score,
        "sparse_score_source": comparison.sparse_score_source,
        "human_normalized_score": comparison.human_normalized_score,
        "human_normalized_score_clipped": comparison.human_normalized_score_clipped,
        "delta_vs_human_score": comparison.delta_vs_human_score,
        "delta_vs_human_normalized": comparison.delta_vs_human_normalized,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "reward_hacking": hacking_payload,
    }
    _emit_payload(payload, json_output)


def _resolve_score_input(
    *,
    label: str,
    report_path: Path | None,
    explicit_score: float | None,
) -> tuple[float, str]:
    """Resolve one required score either from report payload or explicit input."""

    if report_path is not None and explicit_score is not None:
        raise typer.BadParameter(
            f"Provide either --{label}-report or --{label}-score, not both."
        )
    if report_path is not None:
        payload = load_report_payload(report_path)
        score, source = extract_primary_score_from_report(payload)
        return score, f"{report_path}::{source}"
    if explicit_score is not None:
        return float(explicit_score), "explicit_cli_score"
    raise typer.BadParameter(
        f"Missing {label} score: provide --{label}-score or --{label}-report."
    )


def _resolve_probe_scores(
    *,
    probe_reports: list[Path] | None,
    explicit_probe_scores: list[float] | None,
) -> list[float]:
    """Return probe scores merged from explicit values and/or report files."""

    scores: list[float] = []
    if explicit_probe_scores is not None:
        scores.extend(float(item) for item in explicit_probe_scores)
    if probe_reports is not None:
        for report_path in probe_reports:
            payload = load_report_payload(report_path)
            score, _ = extract_primary_score_from_report(payload)
            scores.append(score)
    return scores
