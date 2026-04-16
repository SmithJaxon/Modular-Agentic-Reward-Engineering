"""
Summary: Default tool executors for agentic broker scaffolding.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.factory import resolve_backend_adapter
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.schemas.budget_state import BudgetState
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.schemas.tool_contracts import ToolRequest, ToolResult, ToolResultStatus


def budget_snapshot_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Return the current remaining budget state as a structured tool payload.
    """
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output={"remaining_budget": budget_state.remaining()},
    )


def read_artifact_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Read and return a bounded preview of a local artifact path.
    """
    _ = budget_state
    raw_path = request.arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error="missing required argument: path",
        )
    path = Path(raw_path)
    if not path.exists() or not path.is_file():
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.FAILED,
            error=f"artifact not found: {path}",
        )
    max_chars_raw = request.arguments.get("max_chars", 2000)
    max_chars = int(max_chars_raw) if isinstance(max_chars_raw, int | float) else 2000
    preview = path.read_text(encoding="utf-8", errors="replace")[: max(64, max_chars)]
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output={"path": str(path), "preview": preview},
        artifact_refs=(str(path),),
    )


def run_experiment_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Execute one backend experiment using the normalized adapter interface.
    """
    _ = budget_state
    try:
        environment_id = _required_text(request.arguments, "environment_id")
        backend_name = _required_text(request.arguments, "environment_backend")
        objective_text = _resolve_text_input(
            request.arguments,
            text_key="objective_text",
            file_key="objective_file",
        )
        reward_definition = _resolve_text_input(
            request.arguments,
            text_key="reward_definition",
            file_key="reward_file",
        )
        session_id = _optional_text(request.arguments, "session_id", "agentic-session")
        variant_label = _optional_text(request.arguments, "variant_label", "default")
        iteration_index = _optional_int(request.arguments, "iteration_index", 0, minimum=0)
        seed = _optional_int(request.arguments, "seed", 7)
        overrides = _optional_dict(request.arguments, "overrides")
        include_reflection = _optional_bool(request.arguments, "include_reflection", True)
        candidate_id = _optional_text(request.arguments, "candidate_id", None)
        backend = EnvironmentBackend(backend_name)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error=f"invalid run_experiment arguments: {exc}",
        )

    payload = ExperimentInput(
        session_id=session_id or "agentic-session",
        environment_id=environment_id,
        environment_backend=backend,
        reward_definition=reward_definition,
        iteration_index=iteration_index,
        objective_text=objective_text,
        variant_label=variant_label or "default",
        seed=seed,
        overrides=overrides,
    )
    try:
        adapter = resolve_backend_adapter(backend)
        performance = adapter.run_performance(payload)
        reflection = adapter.run_reflection(payload) if include_reflection else None
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.FAILED,
            error=f"run_experiment execution failed ({exc.__class__.__name__}): {exc}",
        )

    metric_source = performance.metrics
    training_timesteps = _extract_int_metric(metric_source, "total_timesteps")
    evaluation_episodes = _extract_int_metric(metric_source, "evaluation_episodes_consumed")
    resolved_candidate_id = candidate_id or f"cand-{request.request_id[-8:]}"
    output: dict[str, Any] = {
        "candidate_id": resolved_candidate_id,
        "score": performance.score,
        "performance_summary": performance.summary,
        "performance_metrics": performance.metrics,
        "environment_id": environment_id,
        "environment_backend": backend.value,
        "iteration_index": iteration_index,
        "variant_label": variant_label,
        "seed": seed,
    }
    artifact_refs = list(performance.artifact_refs)
    if reflection is not None:
        output["reflection_summary"] = reflection.summary
        output["reflection_metrics"] = reflection.metrics
        for ref in reflection.artifact_refs:
            if ref not in artifact_refs:
                artifact_refs.append(ref)

    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output=output,
        artifact_refs=tuple(artifact_refs),
        model_used=_optional_text(overrides, "llm_model", None),
        training_timesteps=training_timesteps,
        evaluation_episodes=evaluation_episodes,
    )


def run_probe_suite_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Execute robustness probes for one candidate using the shared runner.
    """
    _ = budget_state
    try:
        candidate_id = _required_text(request.arguments, "candidate_id")
        primary_score_raw = request.arguments.get("primary_score")
        if not isinstance(primary_score_raw, int | float):
            raise ValueError("missing required numeric argument: primary_score")
        primary_score = float(primary_score_raw)
        environment_id = _required_text(request.arguments, "environment_id")
        backend_name = _required_text(request.arguments, "environment_backend")
        objective_text = _resolve_text_input(
            request.arguments,
            text_key="objective_text",
            file_key="objective_file",
        )
        reward_definition = _resolve_text_input(
            request.arguments,
            text_key="reward_definition",
            file_key="reward_file",
        )
        session_id = (
            _optional_text(request.arguments, "session_id", "agentic-session")
            or "agentic-session"
        )
        variant_label = _optional_text(request.arguments, "variant_label", "default") or "default"
        iteration_index = _optional_int(request.arguments, "iteration_index", 0, minimum=0)
        seed = _optional_int(request.arguments, "seed", 7)
        overrides = _optional_dict(request.arguments, "overrides")
        backend = EnvironmentBackend(backend_name)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error=f"invalid run_probe_suite arguments: {exc}",
        )

    payload = ExperimentInput(
        session_id=session_id,
        environment_id=environment_id,
        environment_backend=backend,
        reward_definition=reward_definition,
        iteration_index=iteration_index,
        objective_text=objective_text,
        variant_label=variant_label,
        seed=seed,
        overrides=overrides,
    )
    try:
        runner = RobustnessRunner()
        result = runner.run(
            candidate_id=candidate_id,
            payload=payload,
            primary_score=primary_score,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.FAILED,
            error=f"run_probe_suite execution failed ({exc.__class__.__name__}): {exc}",
        )

    assessment = result.analysis.assessment
    artifact_refs: list[str] = []
    training_timesteps = 0
    evaluation_episodes = 0
    runs_payload: list[dict[str, Any]] = []
    for run in result.experiment_runs:
        runs_payload.append(run.model_dump(mode="json"))
        for ref in run.artifact_refs:
            if ref not in artifact_refs:
                artifact_refs.append(ref)
        metrics = run.metrics
        if isinstance(metrics, dict):
            training_timesteps += _extract_int_metric(metrics, "total_timesteps")
            evaluation_episodes += _extract_int_metric(metrics, "evaluation_episodes_consumed")

    output = {
        "candidate_id": candidate_id,
        "assessment": assessment.model_dump(mode="json"),
        "risk_level": assessment.risk_level.value,
        "risk_notes": assessment.risk_notes,
        "degradation_ratio": assessment.degradation_ratio,
        "robustness_bonus": result.analysis.robustness_bonus,
        "tradeoff_rationale": result.analysis.tradeoff_rationale,
        "minor_robustness_risk_accepted": result.analysis.minor_robustness_risk_accepted,
        "probe_runs": runs_payload,
    }
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output=output,
        artifact_refs=tuple(artifact_refs),
        model_used=_optional_text(overrides, "llm_model", None),
        training_timesteps=training_timesteps,
        evaluation_episodes=evaluation_episodes,
    )


def compare_candidates_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Compare candidate snapshots and select the highest aggregate candidate.
    """
    _ = budget_state
    raw_candidates = request.arguments.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error="missing required non-empty argument: candidates",
        )

    normalized: list[dict[str, Any]] = []
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        score = entry.get("score")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if not isinstance(score, int | float):
            continue
        robustness_bonus_raw = entry.get("robustness_bonus", 0.0)
        robustness_bonus = (
            float(robustness_bonus_raw)
            if isinstance(robustness_bonus_raw, int | float)
            else 0.0
        )
        aggregate_score = float(score) + robustness_bonus
        normalized.append(
            {
                "candidate_id": candidate_id,
                "score": float(score),
                "robustness_bonus": robustness_bonus,
                "aggregate_score": aggregate_score,
                "risk_level": entry.get("risk_level", "unknown"),
            }
        )
    if not normalized:
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error="no valid candidate rows found in candidates argument",
        )

    ranking = sorted(normalized, key=lambda row: row["aggregate_score"], reverse=True)
    winner = ranking[0]
    output = {
        "best_candidate_id": winner["candidate_id"],
        "best_aggregate_score": winner["aggregate_score"],
        "ranking": ranking,
        "selection_summary": (
            f"Selected {winner['candidate_id']} with aggregate score "
            f"{winner['aggregate_score']:.3f} from {len(ranking)} candidates."
        ),
    }
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output=output,
    )


def export_report_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Export a provided report payload to disk for downstream consumption.
    """
    _ = budget_state
    report_payload = request.arguments.get("report_payload")
    if not isinstance(report_payload, dict):
        return ToolResult(
            request_id=request.request_id,
            turn_index=request.turn_index,
            tool_name=request.tool_name,
            status=ToolResultStatus.REJECTED,
            error="missing required object argument: report_payload",
        )
    output_path_text = _optional_text(request.arguments, "output_path", None)
    if output_path_text is None:
        output_path_text = f".rewardlab/agentic_exports/{request.request_id}.json"
    output_path = Path(output_path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.COMPLETED,
        output={"report_path": str(output_path)},
        artifact_refs=(str(output_path),),
    )


def not_implemented_executor(request: ToolRequest, budget_state: BudgetState) -> ToolResult:
    """
    Return a structured rejection for tools missing an executor implementation.
    """
    _ = budget_state
    return ToolResult(
        request_id=request.request_id,
        turn_index=request.turn_index,
        tool_name=request.tool_name,
        status=ToolResultStatus.REJECTED,
        error=f"tool {request.tool_name} does not have an executor implementation",
    )


def _required_text(arguments: dict[str, Any], key: str) -> str:
    """
    Resolve a required non-empty text argument.
    """
    raw = arguments.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"missing required argument: {key}")
    return raw.strip()


def _optional_text(arguments: dict[str, Any], key: str, default: str | None) -> str | None:
    """
    Resolve an optional text argument with fallback default.
    """
    raw = arguments.get(key, default)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"invalid text argument: {key}")
    trimmed = raw.strip()
    if not trimmed:
        if default is None:
            return None
        return default
    return trimmed


def _optional_int(
    arguments: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """
    Resolve an optional integer argument with range checks.
    """
    raw = arguments.get(key, default)
    if not isinstance(raw, int | float):
        raise ValueError(f"invalid integer argument: {key}")
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"argument {key} must be >= {minimum}")
    return value


def _optional_bool(arguments: dict[str, Any], key: str, default: bool) -> bool:
    """
    Resolve an optional boolean argument.
    """
    raw = arguments.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"invalid boolean argument: {key}")
    return raw


def _optional_dict(arguments: dict[str, Any], key: str) -> dict[str, Any]:
    """
    Resolve an optional dictionary argument.
    """
    raw = arguments.get(key, {})
    if not isinstance(raw, dict):
        raise ValueError(f"invalid object argument: {key}")
    return dict(raw)


def _resolve_text_input(
    arguments: dict[str, Any],
    *,
    text_key: str,
    file_key: str,
) -> str:
    """
    Resolve a text input from inline text or a local file path.
    """
    inline = arguments.get(text_key)
    if isinstance(inline, str) and inline.strip():
        return inline.strip()
    raw_file = arguments.get(file_key)
    if not isinstance(raw_file, str) or not raw_file.strip():
        raise ValueError(f"missing one of {text_key} or {file_key}")
    path = Path(raw_file)
    if not path.exists() or not path.is_file():
        raise ValueError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"file {path} is empty")
    return text


def _extract_int_metric(metrics: dict[str, Any], key: str) -> int:
    """
    Extract a non-negative integer metric from experiment output when present.
    """
    raw = metrics.get(key, 0)
    if not isinstance(raw, int | float):
        return 0
    return max(0, int(raw))
