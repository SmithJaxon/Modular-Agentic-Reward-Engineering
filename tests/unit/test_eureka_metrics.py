"""
Summary: Unit tests for Eureka-comparable scoring utilities.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

import pytest

from rewardlab.agentic.eureka_metrics import (
    compute_eureka_comparison_metrics,
    compute_human_normalized_score,
    compute_reward_hacking_metrics,
    extract_primary_score_from_report,
)


def test_compute_human_normalized_score_matches_expected_formula() -> None:
    """Human normalized score should match the method-sparse over |human-sparse| form."""

    score = compute_human_normalized_score(
        method_score=0.52,
        human_score=0.47,
        sparse_score=0.20,
    )
    assert score == pytest.approx((0.52 - 0.20) / abs(0.47 - 0.20), rel=1e-9, abs=1e-9)


def test_compute_eureka_comparison_metrics_returns_raw_and_clipped_hns() -> None:
    """Comparison metrics should expose raw HNS, clipped HNS, and human deltas."""

    metrics = compute_eureka_comparison_metrics(
        method_score=1.50,
        human_score=1.00,
        sparse_score=0.00,
        method_score_source="method",
        human_score_source="human",
        sparse_score_source="sparse",
        clip_min=0.0,
        clip_max=1.0,
    )

    assert metrics.human_normalized_score == pytest.approx(1.5, rel=1e-9, abs=1e-9)
    assert metrics.human_normalized_score_clipped == pytest.approx(1.0, rel=1e-9, abs=1e-9)
    assert metrics.delta_vs_human_score == pytest.approx(0.5, rel=1e-9, abs=1e-9)
    assert metrics.delta_vs_human_normalized == pytest.approx(0.5, rel=1e-9, abs=1e-9)


def test_extract_primary_score_prefers_final_eval_mean_from_report() -> None:
    """Report extraction should prioritize final_eval_mean_score when present."""

    payload = {
        "experiment": {
            "best_candidate_id": "exp-candidate-002",
            "metadata": {"final_eval_mean_score": 0.491275},
        },
        "candidates": [
            {"candidate_id": "exp-candidate-002", "aggregate_score": 0.523361},
        ],
    }
    score, source = extract_primary_score_from_report(payload)
    assert score == pytest.approx(0.491275, rel=1e-9, abs=1e-9)
    assert source == "experiment.metadata.final_eval_mean_score"


def test_compute_reward_hacking_metrics_reports_expected_trends() -> None:
    """Probe metrics should reflect degradation and assign a non-low risk when large."""

    metrics = compute_reward_hacking_metrics(
        method_score=0.60,
        human_score=0.50,
        sparse_score=0.20,
        probe_scores=[0.58, 0.44, 0.21],
    )

    assert metrics.probe_count == 3
    assert metrics.probe_min_score == pytest.approx(0.21, rel=1e-9, abs=1e-9)
    assert metrics.worst_score_degradation_ratio == pytest.approx(0.65, rel=1e-9, abs=1e-9)
    assert metrics.worst_human_normalized_drop > 0.0
    expected_relative_performance = round(((0.58 + 0.44 + 0.21) / 3) / 0.60, 6)
    assert metrics.perils_relative_reward_function_performance == pytest.approx(
        expected_relative_performance,
        rel=1e-9,
        abs=1e-9,
    )
    expected_hacking_severity = round(1.0 - min(max(expected_relative_performance, 0.0), 1.0), 6)
    assert metrics.perils_hacking_severity == pytest.approx(
        expected_hacking_severity,
        rel=1e-9,
        abs=1e-9,
    )
    assert metrics.hacking_risk_index > 0.2
    assert metrics.hacking_risk_level in {"medium", "high"}
