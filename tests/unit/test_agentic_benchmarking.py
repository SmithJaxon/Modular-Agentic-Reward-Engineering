"""
Summary: Unit tests for autonomous experiment benchmark summarization.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

import pytest

from rewardlab.agentic.benchmarking import (
    BenchmarkRunSummary,
    aggregate_benchmark_summaries,
    summarize_trace_for_benchmark,
)


def test_summarize_trace_for_benchmark_extracts_core_metrics() -> None:
    """Trace summarizer should compute score deltas, actions, and budget usage."""

    trace_payload = {
        "experiment": {
            "experiment_id": "exp-001",
            "status": "completed",
            "stop_reason": "plateau_detected",
            "best_candidate_id": "exp-001-candidate-001",
            "created_at": "2026-04-11T00:00:00Z",
            "ended_at": "2026-04-11T00:30:00Z",
            "budget_ledger": {
                "consumed_total_tokens": 1200,
                "consumed_total_usd": 0.0,
                "consumed_experiments": 2,
                "consumed_train_timesteps": 100000,
                "consumed_human_feedback_requests": 0,
            },
        },
        "candidates": [
            {"iteration_index": 0, "aggregate_score": 0.40},
            {"iteration_index": 1, "aggregate_score": 0.44},
            {"iteration_index": 2, "aggregate_score": 0.42},
        ],
        "decisions": [
            {"action_type": "run_experiment"},
            {"action_type": "propose_reward"},
            {"action_type": "run_experiment"},
        ],
    }

    summary = summarize_trace_for_benchmark(seed=7, trace_payload=trace_payload)

    assert summary.experiment_id == "exp-001"
    assert summary.seed == 7
    assert summary.best_score == 0.44
    assert summary.baseline_score == 0.40
    assert summary.final_score == 0.42
    assert summary.improvement_absolute == pytest.approx(0.04, rel=1e-9, abs=1e-9)
    assert summary.decision_count == 3
    assert summary.action_counts["run_experiment"] == 2
    assert summary.elapsed_minutes == 30.0


def test_aggregate_benchmark_summaries_reports_overview_and_action_mix() -> None:
    """Aggregate summary should include reliability, score, and action metrics."""

    summaries = [
        BenchmarkRunSummary(
            experiment_id="exp-a",
            seed=1,
            status="completed",
            stop_reason="plateau_detected",
            best_candidate_id="a-1",
            baseline_score=0.40,
            final_score=0.45,
            best_score=0.45,
            best_iteration_index=1,
            improvement_absolute=0.05,
            improvement_relative=0.125,
            decision_count=4,
            action_counts={"run_experiment": 2, "propose_reward": 2},
            consumed_total_tokens=1000,
            consumed_total_usd=0.0,
            consumed_experiments=2,
            consumed_train_timesteps=100000,
            consumed_human_feedback_requests=0,
            elapsed_minutes=20.0,
        ),
        BenchmarkRunSummary(
            experiment_id="exp-b",
            seed=2,
            status="completed",
            stop_reason="no_improve_streak_reached",
            best_candidate_id="b-1",
            baseline_score=0.41,
            final_score=0.40,
            best_score=0.41,
            best_iteration_index=0,
            improvement_absolute=0.0,
            improvement_relative=0.0,
            decision_count=3,
            action_counts={"run_experiment": 2, "propose_reward": 1},
            consumed_total_tokens=800,
            consumed_total_usd=0.0,
            consumed_experiments=2,
            consumed_train_timesteps=100000,
            consumed_human_feedback_requests=0,
            elapsed_minutes=18.0,
        ),
    ]

    aggregate = aggregate_benchmark_summaries(summaries)

    overview = aggregate["overview"]
    assert isinstance(overview, dict)
    assert overview["run_count"] == 2
    assert overview["completed_count"] == 2
    assert overview["improved_count"] == 1

    decisions = aggregate["decision_metrics"]
    assert isinstance(decisions, dict)
    action_totals = decisions["action_totals"]
    assert isinstance(action_totals, dict)
    assert action_totals["run_experiment"] == 4
    assert action_totals["propose_reward"] == 3
