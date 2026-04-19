"""
Summary: Typer command handlers for autonomous agent experiment operations.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, List, Optional

try:
    from typing import Annotated
except ImportError:  # pragma: no cover - Python <3.9 compatibility
    from typing_extensions import Annotated

import typer

from rewardlab.agentic.eureka_metrics import (
    compute_eureka_comparison_metrics,
    compute_reward_hacking_metrics,
    extract_primary_score_from_report,
    load_report_payload,
)
from rewardlab.agentic.spec_loader import load_experiment_spec
from rewardlab.agentic.service import AgentExperimentService
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.experiments.isaacgym_runner import (
    IsaacGymSubprocessConfig,
    resolve_worker_command,
)
from rewardlab.schemas.agent_experiment import ExecutionIsaacConfig
from rewardlab.schemas.session_config import EnvironmentBackend

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
    experiment_id: Annotated[Optional[str], typer.Option()] = None,
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
    score: Annotated[Optional[float], typer.Option()] = None,
    request_id: Annotated[Optional[str], typer.Option()] = None,
    artifact_ref: Annotated[Optional[str], typer.Option()] = None,
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
    seed: Annotated[Optional[List[int]], typer.Option("--seed")] = None,
    benchmark_id: Annotated[Optional[str], typer.Option()] = None,
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
        Optional[Path], typer.Option("--method-report", exists=True, dir_okay=False)
    ] = None,
    method_score: Annotated[Optional[float], typer.Option("--method-score")] = None,
    human_report: Annotated[
        Optional[Path], typer.Option("--human-report", exists=True, dir_okay=False)
    ] = None,
    human_score: Annotated[Optional[float], typer.Option("--human-score")] = None,
    sparse_report: Annotated[
        Optional[Path], typer.Option("--sparse-report", exists=True, dir_okay=False)
    ] = None,
    sparse_score: Annotated[Optional[float], typer.Option("--sparse-score")] = None,
    probe_report: Annotated[
        Optional[List[Path]], typer.Option("--probe-report", exists=True, dir_okay=False)
    ] = None,
    probe_score: Annotated[Optional[List[float]], typer.Option("--probe-score")] = None,
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
    hacking_payload: Optional[dict[str, Any]] = None
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


@experiment_app.command("runtime-check")
def runtime_check(
    backend: EnvironmentBackend = typer.Option(EnvironmentBackend.ISAAC_GYM, "--backend"),
    environment_id: Optional[List[str]] = typer.Option(None, "--environment-id"),
    file: Optional[Path] = typer.Option(None, "--file", exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run backend runtime preflight checks before queueing full experiments."""

    if backend != EnvironmentBackend.ISAAC_GYM:
        payload = {
            "backend": backend.value,
            "status": "unsupported",
            "reason": "runtime-check currently supports --backend isaacgym only",
        }
        _emit_payload(payload, json_output)
        raise typer.Exit(code=1)

    isaac_config = ExecutionIsaacConfig()
    if file is not None:
        spec = load_experiment_spec(file)
        if backend != spec.environment.backend:
            payload = {
                "backend": backend.value,
                "status": "error",
                "reason": (
                    f"runtime-check backend ({backend.value}) does not match "
                    f"spec backend ({spec.environment.backend.value})"
                ),
            }
            _emit_payload(payload, json_output)
            raise typer.Exit(code=1)
        isaac_config = spec.execution.isaac
        if environment_id is None:
            environment_id = [spec.environment.id]

    payload = _isaac_runtime_check_payload(
        environment_ids=environment_id,
        isaac_config=isaac_config,
    )
    _emit_payload(payload, json_output)
    if payload.get("status") != "ok":
        raise typer.Exit(code=1)


def _resolve_score_input(
    *,
    label: str,
    report_path: Optional[Path],
    explicit_score: Optional[float],
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
    probe_reports: Optional[List[Path]],
    explicit_probe_scores: Optional[List[float]],
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


def _isaac_runtime_check_payload(
    *,
    environment_ids: Optional[List[str]],
    isaac_config: ExecutionIsaacConfig,
) -> dict[str, Any]:
    """Collect a structured Isaac runtime preflight report."""

    task_ids = environment_ids if environment_ids and len(environment_ids) > 0 else [
        "Cartpole",
        "Humanoid",
        "AllegroHand",
    ]
    backend = IsaacGymBackend(cfg_dir_override=isaac_config.cfg_dir)
    statuses = {
        task_id: backend.get_runtime_status(task_id).model_dump(mode="json")
        for task_id in task_ids
    }
    available_tasks = sorted(backend.list_available_tasks())
    config_dir = backend.resolve_config_dir()
    import_status = _collect_isaac_import_status()
    worker_command = resolve_worker_command(
        IsaacGymSubprocessConfig(worker_command=isaac_config.worker_command)
    )
    all_ready = all(bool(status.get("ready")) for status in statuses.values())
    return {
        "backend": EnvironmentBackend.ISAAC_GYM.value,
        "status": "ok" if all_ready else "error",
        "python_executable": sys.executable,
        "checks": {
            "task_status": statuses,
            "available_tasks": available_tasks,
            "config_dir": config_dir,
            "worker_command": worker_command,
            "imports": import_status,
        },
    }


def _collect_isaac_import_status() -> dict[str, Any]:
    """Return import/cuda details relevant to Isaac runtime validation."""

    payload: dict[str, Any] = {}
    try:
        import torch  # type: ignore[import-not-found]

        payload["torch_importable"] = True
        payload["torch_version"] = getattr(torch, "__version__", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_importable"] = False
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
        payload["cuda_available"] = False
        payload["cuda_device_count"] = 0

    try:
        import isaacgym  # noqa: F401  # type: ignore[import-not-found]

        payload["isaacgym_importable"] = True
    except Exception as exc:
        payload["isaacgym_importable"] = False
        payload["isaacgym_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import isaacgymenvs  # noqa: F401  # type: ignore[import-not-found]

        payload["isaacgymenvs_importable"] = True
    except Exception as exc:
        payload["isaacgymenvs_importable"] = False
        payload["isaacgymenvs_error"] = f"{type(exc).__name__}: {exc}"
    return payload
