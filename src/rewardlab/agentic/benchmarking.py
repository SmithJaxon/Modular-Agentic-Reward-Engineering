"""
Summary: Benchmark summarization utilities for autonomous agent experiments.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from statistics import mean, median, stdev
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunSummary:
    """Normalized benchmark summary for one experiment run."""

    experiment_id: str
    seed: int
    status: str
    stop_reason: str | None
    best_candidate_id: str | None
    baseline_score: float | None
    final_score: float | None
    best_score: float | None
    best_iteration_index: int | None
    improvement_absolute: float | None
    improvement_relative: float | None
    decision_count: int
    action_counts: dict[str, int]
    consumed_total_tokens: int
    consumed_total_usd: float
    consumed_experiments: int
    consumed_train_timesteps: int
    consumed_human_feedback_requests: int
    elapsed_minutes: float | None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload for one run summary."""

        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "best_candidate_id": self.best_candidate_id,
            "baseline_score": self.baseline_score,
            "final_score": self.final_score,
            "best_score": self.best_score,
            "best_iteration_index": self.best_iteration_index,
            "improvement_absolute": self.improvement_absolute,
            "improvement_relative": self.improvement_relative,
            "decision_count": self.decision_count,
            "action_counts": dict(self.action_counts),
            "consumed_total_tokens": self.consumed_total_tokens,
            "consumed_total_usd": self.consumed_total_usd,
            "consumed_experiments": self.consumed_experiments,
            "consumed_train_timesteps": self.consumed_train_timesteps,
            "consumed_human_feedback_requests": self.consumed_human_feedback_requests,
            "elapsed_minutes": self.elapsed_minutes,
        }


def summarize_trace_for_benchmark(
    *,
    seed: int,
    trace_payload: dict[str, object],
) -> BenchmarkRunSummary:
    """Build a run summary from a full experiment trace payload."""

    experiment = _as_mapping(trace_payload.get("experiment"))
    candidates = [_as_mapping(item) for item in _as_sequence(trace_payload.get("candidates"))]
    decisions = [_as_mapping(item) for item in _as_sequence(trace_payload.get("decisions"))]
    ledger = _as_mapping(experiment.get("budget_ledger"))

    sorted_candidates = sorted(
        candidates,
        key=lambda item: _int_or_default(item.get("iteration_index"), default=0),
    )
    scored = [
        item
        for item in sorted_candidates
        if isinstance(item.get("aggregate_score"), (int, float))
    ]

    baseline_score = _score_for_iteration(sorted_candidates, 0)
    final_score = _optional_float(scored[-1].get("aggregate_score")) if scored else None
    best_candidate = (
        max(
            scored,
            key=lambda item: _optional_float(item.get("aggregate_score")) or float("-inf"),
        )
        if scored
        else None
    )
    best_score = (
        _optional_float(best_candidate.get("aggregate_score"))
        if best_candidate is not None
        else None
    )
    best_iteration_index = (
        _int_or_default(best_candidate.get("iteration_index"), default=0)
        if best_candidate is not None
        else None
    )
    improvement_absolute = (
        (best_score - baseline_score)
        if best_score is not None and baseline_score is not None
        else None
    )
    improvement_relative = (
        ((best_score - baseline_score) / abs(baseline_score))
        if best_score is not None
        and baseline_score is not None
        and abs(baseline_score) > 1e-9
        else None
    )

    action_counts = Counter(
        str(item.get("action_type"))
        for item in decisions
        if isinstance(item.get("action_type"), str)
    )
    elapsed_minutes = _elapsed_minutes(
        created_at=experiment.get("created_at"),
        ended_at=experiment.get("ended_at"),
    )
    return BenchmarkRunSummary(
        experiment_id=str(experiment.get("experiment_id", "")),
        seed=seed,
        status=str(experiment.get("status", "")),
        stop_reason=(
            str(experiment.get("stop_reason"))
            if isinstance(experiment.get("stop_reason"), str)
            else None
        ),
        best_candidate_id=(
            str(experiment.get("best_candidate_id"))
            if isinstance(experiment.get("best_candidate_id"), str)
            else None
        ),
        baseline_score=baseline_score,
        final_score=final_score,
        best_score=best_score,
        best_iteration_index=best_iteration_index,
        improvement_absolute=improvement_absolute,
        improvement_relative=improvement_relative,
        decision_count=len(decisions),
        action_counts=dict(action_counts),
        consumed_total_tokens=_int_value(ledger.get("consumed_total_tokens")),
        consumed_total_usd=_float_value(ledger.get("consumed_total_usd")),
        consumed_experiments=_int_value(ledger.get("consumed_experiments")),
        consumed_train_timesteps=_int_value(ledger.get("consumed_train_timesteps")),
        consumed_human_feedback_requests=_int_value(
            ledger.get("consumed_human_feedback_requests")
        ),
        elapsed_minutes=elapsed_minutes,
    )


def aggregate_benchmark_summaries(
    summaries: list[BenchmarkRunSummary],
) -> dict[str, object]:
    """Build aggregate benchmark metrics across run summaries."""

    run_count = len(summaries)
    completed_count = sum(1 for item in summaries if item.status == "completed")
    improved_count = sum(
        1
        for item in summaries
        if item.improvement_absolute is not None and item.improvement_absolute > 0
    )
    non_degraded_count = sum(
        1
        for item in summaries
        if item.improvement_absolute is not None and item.improvement_absolute >= 0
    )

    stop_reasons = Counter(
        item.stop_reason for item in summaries if item.stop_reason is not None
    )
    action_totals: Counter[str] = Counter()
    for item in summaries:
        action_totals.update(item.action_counts)
    total_actions = sum(action_totals.values())
    action_mix = {
        action: {
            "count": count,
            "share": (count / total_actions) if total_actions > 0 else 0.0,
        }
        for action, count in sorted(action_totals.items())
    }

    baseline_scores = _collect_numeric(summaries, "baseline_score")
    final_scores = _collect_numeric(summaries, "final_score")
    best_scores = _collect_numeric(summaries, "best_score")
    improvement_absolute = _collect_numeric(summaries, "improvement_absolute")
    improvement_relative = _collect_numeric(summaries, "improvement_relative")
    elapsed_minutes = _collect_numeric(summaries, "elapsed_minutes")

    tokens_used = [float(item.consumed_total_tokens) for item in summaries]
    usd_used = [float(item.consumed_total_usd) for item in summaries]
    experiments_used = [float(item.consumed_experiments) for item in summaries]
    timesteps_used = [float(item.consumed_train_timesteps) for item in summaries]
    decisions_used = [float(item.decision_count) for item in summaries]

    efficiency_improvement_per_1k_tokens = []
    for item in summaries:
        if item.improvement_absolute is None:
            continue
        if item.consumed_total_tokens <= 0:
            continue
        efficiency_improvement_per_1k_tokens.append(
            item.improvement_absolute / (item.consumed_total_tokens / 1000.0)
        )

    best_run = _best_run_by_score(summaries)
    return {
        "overview": {
            "run_count": run_count,
            "completed_count": completed_count,
            "completed_rate": _safe_rate(completed_count, run_count),
            "improved_count": improved_count,
            "improved_rate": _safe_rate(improved_count, run_count),
            "non_degraded_count": non_degraded_count,
            "non_degraded_rate": _safe_rate(non_degraded_count, run_count),
            "best_experiment_id": best_run.experiment_id if best_run is not None else None,
            "best_score": best_run.best_score if best_run is not None else None,
        },
        "score_metrics": {
            "baseline_score": _stat_block(baseline_scores),
            "final_score": _stat_block(final_scores),
            "best_score": _stat_block(best_scores),
            "improvement_absolute": _stat_block(improvement_absolute),
            "improvement_relative": _stat_block(improvement_relative),
        },
        "resource_metrics": {
            "consumed_total_tokens": _stat_block(tokens_used),
            "consumed_total_usd": _stat_block(usd_used),
            "consumed_experiments": _stat_block(experiments_used),
            "consumed_train_timesteps": _stat_block(timesteps_used),
            "decision_count": _stat_block(decisions_used),
            "elapsed_minutes": _stat_block(elapsed_minutes),
            "improvement_per_1k_tokens": _stat_block(efficiency_improvement_per_1k_tokens),
        },
        "decision_metrics": {
            "action_totals": dict(action_totals),
            "action_mix": action_mix,
            "stop_reason_counts": dict(stop_reasons),
        },
    }


def _best_run_by_score(summaries: list[BenchmarkRunSummary]) -> BenchmarkRunSummary | None:
    """Return the run with highest best_score, if any."""

    scored = [item for item in summaries if item.best_score is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: float(item.best_score or float("-inf")))


def _score_for_iteration(
    candidates: list[dict[str, object]],
    iteration_index: int,
) -> float | None:
    """Return candidate score for one iteration index when available."""

    for candidate in candidates:
        if _int_or_default(candidate.get("iteration_index"), default=-1) != iteration_index:
            continue
        score = _optional_float(candidate.get("aggregate_score"))
        if score is not None:
            return score
    return None


def _elapsed_minutes(*, created_at: object, ended_at: object) -> float | None:
    """Return elapsed wall-clock minutes from serialized timestamps."""

    if not isinstance(created_at, str) or not isinstance(ended_at, str):
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((ended - created).total_seconds() / 60.0, 0.0)


def _collect_numeric(summaries: list[BenchmarkRunSummary], key: str) -> list[float]:
    """Collect numeric values for one summary attribute."""

    values: list[float] = []
    for item in summaries:
        value = getattr(item, key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _stat_block(values: list[float]) -> dict[str, float | int | None]:
    """Return robust descriptive stats for numeric values."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stddev": None,
            "sem": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    count = len(values)
    result: dict[str, float | int | None] = {
        "count": count,
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "stddev": None,
        "sem": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if count < 2:
        return result

    std = stdev(values)
    sem = std / sqrt(count)
    mu = float(result["mean"] or 0.0)
    ci = 1.96 * sem
    result.update(
        {
            "stddev": std,
            "sem": sem,
            "ci95_low": mu - ci,
            "ci95_high": mu + ci,
        }
    )
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate for two integer counters."""

    if denominator <= 0:
        return 0.0
    return max(min(numerator / denominator, 1.0), 0.0)


def _as_mapping(value: object) -> dict[str, object]:
    """Return mapping value as dict or empty mapping."""

    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_sequence(value: object) -> list[object]:
    """Return sequence value as list or empty list."""

    if isinstance(value, list):
        return list(value)
    return []


def _int_value(value: object) -> int:
    """Return coerced non-negative integer from loosely typed value."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _float_value(value: object) -> float:
    """Return coerced non-negative float from loosely typed value."""

    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    return 0.0


def _int_or_default(value: object, *, default: int) -> int:
    """Return integer value from object or fallback default."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _optional_float(value: object) -> float | None:
    """Return float value when object is numeric-like."""

    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
