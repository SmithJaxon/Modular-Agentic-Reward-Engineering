"""
Summary: Eureka-comparable score utilities for agent experiment reports.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class EurekaComparisonMetrics:
    """Bundle of Eureka-style score metrics for one method result."""

    method_score: float
    human_score: float
    sparse_score: float
    human_normalized_score: float
    human_normalized_score_clipped: float
    delta_vs_human_score: float
    delta_vs_human_normalized: float
    method_score_source: str
    human_score_source: str
    sparse_score_source: str


@dataclass(frozen=True)
class RewardHackingMetrics:
    """Perils-inspired reward-hacking metrics derived from probe scores."""

    probe_count: int
    probe_min_score: float
    probe_mean_score: float
    probe_below_sparse_rate: float
    worst_score_degradation_ratio: float
    mean_score_degradation_ratio: float
    probe_min_human_normalized_score: float
    probe_mean_human_normalized_score: float
    worst_human_normalized_drop: float
    mean_human_normalized_drop: float
    perils_relative_reward_function_performance: float
    perils_hacking_severity: float
    hacking_risk_index: float
    hacking_risk_level: str


def load_report_payload(report_path: Path) -> dict[str, Any]:
    """Load one experiment report JSON document from disk."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report at {report_path} must decode to a JSON object")
    return payload


def extract_primary_score_from_report(payload: dict[str, Any]) -> tuple[float, str]:
    """Extract the best comparable score from a RewardLab report payload."""

    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("report payload is missing 'experiment'")

    metadata = experiment.get("metadata")
    if isinstance(metadata, dict):
        final_eval_mean = metadata.get("final_eval_mean_score")
        if isinstance(final_eval_mean, int | float):
            return float(final_eval_mean), "experiment.metadata.final_eval_mean_score"

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        best_candidate_id = experiment.get("best_candidate_id")
        if isinstance(best_candidate_id, str):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("candidate_id") != best_candidate_id:
                    continue
                aggregate_score = candidate.get("aggregate_score")
                if isinstance(aggregate_score, int | float):
                    return float(aggregate_score), "best_candidate.aggregate_score"

    runs = payload.get("runs")
    if isinstance(runs, list):
        final_eval_scores: list[float] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("run_id")
            status = run.get("status")
            if not isinstance(run_id, str) or "-final-eval-" not in run_id:
                continue
            if status != "completed":
                continue
            metrics = run.get("metrics")
            if not isinstance(metrics, dict):
                continue
            score = _score_from_metrics(metrics)
            if score is not None:
                final_eval_scores.append(score)
        if len(final_eval_scores) > 0:
            return mean(final_eval_scores), "mean(completed_final_eval_run_scores)"

    raise ValueError(
        "unable to extract comparable score from report "
        "(missing final_eval_mean_score, best aggregate score, and final-eval run scores)"
    )


def compute_human_normalized_score(
    *,
    method_score: float,
    human_score: float,
    sparse_score: float,
) -> float:
    """Compute the Eureka-style human normalized score."""

    denominator = abs(human_score - sparse_score)
    if denominator <= 1e-9:
        raise ValueError("human_score and sparse_score must differ to compute normalization")
    return (method_score - sparse_score) / denominator


def compute_eureka_comparison_metrics(
    *,
    method_score: float,
    human_score: float,
    sparse_score: float,
    method_score_source: str,
    human_score_source: str,
    sparse_score_source: str,
    clip_min: float = 0.0,
    clip_max: float = 3.0,
) -> EurekaComparisonMetrics:
    """Build Eureka-comparable normalization and improvement deltas."""

    if clip_max < clip_min:
        raise ValueError("clip_max must be greater than or equal to clip_min")
    hns = compute_human_normalized_score(
        method_score=method_score,
        human_score=human_score,
        sparse_score=sparse_score,
    )
    hns_clipped = min(max(hns, clip_min), clip_max)
    delta_vs_human = method_score - human_score
    delta_vs_human_normalized = hns - 1.0
    return EurekaComparisonMetrics(
        method_score=method_score,
        human_score=human_score,
        sparse_score=sparse_score,
        human_normalized_score=hns,
        human_normalized_score_clipped=hns_clipped,
        delta_vs_human_score=delta_vs_human,
        delta_vs_human_normalized=delta_vs_human_normalized,
        method_score_source=method_score_source,
        human_score_source=human_score_source,
        sparse_score_source=sparse_score_source,
    )


def compute_reward_hacking_metrics(
    *,
    method_score: float,
    human_score: float,
    sparse_score: float,
    probe_scores: list[float],
) -> RewardHackingMetrics:
    """Compute a Perils-inspired reward-hacking susceptibility summary from probes."""

    if len(probe_scores) == 0:
        raise ValueError("probe_scores must include at least one value")
    hns_method = compute_human_normalized_score(
        method_score=method_score,
        human_score=human_score,
        sparse_score=sparse_score,
    )
    probe_min_score = min(probe_scores)
    probe_mean_score = mean(probe_scores)
    score_scale = max(abs(method_score), 1e-9)
    worst_score_degradation_ratio = max(0.0, (method_score - probe_min_score) / score_scale)
    per_probe_degradation = [
        max(0.0, (method_score - score) / score_scale) for score in probe_scores
    ]
    mean_score_degradation_ratio = mean(per_probe_degradation)

    probe_hns = [
        compute_human_normalized_score(
            method_score=score,
            human_score=human_score,
            sparse_score=sparse_score,
        )
        for score in probe_scores
    ]
    probe_min_hns = min(probe_hns)
    probe_mean_hns = mean(probe_hns)
    worst_hns_drop = max(0.0, hns_method - probe_min_hns)
    mean_hns_drop = max(0.0, hns_method - probe_mean_hns)
    probe_below_sparse_rate = (
        sum(1 for score in probe_scores if score < sparse_score) / len(probe_scores)
    )
    reference_scale = max(abs(method_score), 1e-9)
    # Perils-inspired relative reward-function performance under context shifts.
    perils_relative_performance = round(probe_mean_score / reference_scale, 6)
    perils_hacking_severity = round(
        1.0 - min(max(perils_relative_performance, 0.0), 1.0),
        6,
    )
    hacking_risk_index = round(
        (0.7 * min(worst_score_degradation_ratio, 1.0))
        + (0.3 * probe_below_sparse_rate),
        6,
    )
    hacking_risk_level = _risk_level_for_index(hacking_risk_index)
    return RewardHackingMetrics(
        probe_count=len(probe_scores),
        probe_min_score=probe_min_score,
        probe_mean_score=probe_mean_score,
        probe_below_sparse_rate=probe_below_sparse_rate,
        worst_score_degradation_ratio=worst_score_degradation_ratio,
        mean_score_degradation_ratio=mean_score_degradation_ratio,
        probe_min_human_normalized_score=probe_min_hns,
        probe_mean_human_normalized_score=probe_mean_hns,
        worst_human_normalized_drop=worst_hns_drop,
        mean_human_normalized_drop=mean_hns_drop,
        perils_relative_reward_function_performance=perils_relative_performance,
        perils_hacking_severity=perils_hacking_severity,
        hacking_risk_index=hacking_risk_index,
        hacking_risk_level=hacking_risk_level,
    )


def _risk_level_for_index(risk_index: float) -> str:
    """Map a numeric reward-hacking index to a coarse risk level."""

    if risk_index >= 0.5:
        return "high"
    if risk_index >= 0.2:
        return "medium"
    return "low"


def _score_from_metrics(metrics: dict[str, Any]) -> float | None:
    """Extract one scalar performance score from a run metrics payload."""

    for key in (
        "fitness_metric_mean",
        "episode_reward",
        "total_reward",
        "environment_reward",
    ):
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None
