"""
Summary: Foundational unit tests for schemas, retry policy, and state transitions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.feedback.gating import evaluate_feedback_gate, summarize_feedback_entries
from rewardlab.orchestrator.state_machine import (
    SessionLifecycleState,
    can_transition,
    ensure_transition,
)
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.experiment_run import (
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentRunType,
)
from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSource
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.schemas.session_report import (
    BestCandidateReport,
    IterationReport,
    RiskLevel,
    SessionReport,
    SessionStatus,
    StopReason,
)
from rewardlab.utils.retry import RetryError, RetryPolicy, run_with_retry


def test_session_config_accepts_valid_payload() -> None:
    """
    Validate that session config accepts complete, valid fields.
    """
    model = SessionConfig(
        objective_text="maximize stability",
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=10,
        feedback_gate=FeedbackGate.ONE_REQUIRED,
    )
    assert model.environment_backend is EnvironmentBackend.GYMNASIUM


def test_session_config_rejects_invalid_iteration_bounds() -> None:
    """
    Validate max_iterations cannot be smaller than no_improve_limit.
    """
    with pytest.raises(ValueError):
        SessionConfig(
            objective_text="maximize stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=5,
            max_iterations=3,
            feedback_gate=FeedbackGate.ONE_REQUIRED,
        )


def test_session_report_requires_iteration_entries() -> None:
    """
    Validate report schema requires at least one iteration summary.
    """
    with pytest.raises(ValueError):
        SessionReport(
            session_id="session-1",
            status=SessionStatus.COMPLETED,
            stop_reason=StopReason.ITERATION_CAP,
            environment_backend=EnvironmentBackend.GYMNASIUM,
            best_candidate=BestCandidateReport(
                candidate_id="cand-1",
                aggregate_score=0.8,
                selection_summary="solid",
            ),
            iterations=[],
        )


def test_session_report_accepts_valid_payload() -> None:
    """
    Validate report schema accepts well-formed payloads.
    """
    report = SessionReport(
        session_id="session-1",
        status=SessionStatus.COMPLETED,
        stop_reason=StopReason.CONVERGENCE,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=BestCandidateReport(
            candidate_id="cand-1",
            aggregate_score=0.85,
            selection_summary="improved stability",
            minor_robustness_risk_accepted=False,
        ),
        iterations=[
            IterationReport(
                iteration_index=0,
                candidate_id="cand-1",
                performance_summary="baseline pass",
                risk_level=RiskLevel.LOW,
                feedback_count=1,
            )
        ],
    )
    assert report.best_candidate.candidate_id == "cand-1"


def test_session_report_accepts_budget_cap_stop_reason() -> None:
    """
    Validate report schema accepts adaptive-budget termination.
    """
    report = SessionReport(
        session_id="session-1",
        status=SessionStatus.COMPLETED,
        stop_reason=StopReason.BUDGET_CAP,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=BestCandidateReport(
            candidate_id="cand-1",
            aggregate_score=0.81,
            selection_summary="Stopped after adaptive budget exhaustion.",
        ),
        iterations=[
            IterationReport(
                iteration_index=0,
                candidate_id="cand-1",
                performance_summary="iteration 0 score=0.810",
                risk_level=RiskLevel.LOW,
                feedback_count=0,
            )
        ],
    )
    assert report.stop_reason is StopReason.BUDGET_CAP


def test_session_report_accepts_running_payload_without_stop_reason() -> None:
    """
    Validate running report exports can omit stop reason until termination.
    """
    report = SessionReport(
        session_id="session-1",
        status=SessionStatus.RUNNING,
        stop_reason=None,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=BestCandidateReport(
            candidate_id="cand-1",
            aggregate_score=0.75,
            selection_summary="pending final gate satisfaction",
        ),
        iterations=[
            IterationReport(
                iteration_index=0,
                candidate_id="cand-1",
                performance_summary="iteration 0 score=0.750",
                risk_level=RiskLevel.LOW,
                feedback_count=0,
            )
        ],
    )
    assert report.stop_reason is None


def test_feedback_entry_rejects_blank_comment() -> None:
    """
    Validate feedback entries reject whitespace-only comments.
    """
    with pytest.raises(ValueError):
        FeedbackEntry(
            feedback_id="fb-1",
            candidate_id="cand-1",
            source_type=FeedbackSource.HUMAN,
            score=0.2,
            comment="   ",
            artifact_ref="demo://session/candidate/latest",
            created_at="2026-04-02T00:00:00+00:00",
        )


def test_feedback_gate_evaluator_tracks_missing_sources() -> None:
    """
    Validate gate evaluation distinguishes one-channel and both-channel requirements.
    """
    human_only = summarize_feedback_entries(
        "cand-1",
        [
            FeedbackEntry(
                feedback_id="fb-1",
                candidate_id="cand-1",
                source_type=FeedbackSource.HUMAN,
                score=0.2,
                comment="Looks aligned.",
                artifact_ref="demo://session/candidate/latest",
                created_at="2026-04-02T00:00:00+00:00",
            )
        ],
    )
    assert evaluate_feedback_gate(FeedbackGate.ONE_REQUIRED, human_only).satisfied is True
    both_required = evaluate_feedback_gate(FeedbackGate.BOTH_REQUIRED, human_only)
    assert both_required.satisfied is False
    assert both_required.missing_sources == (FeedbackSource.PEER,)


def test_experiment_run_rejects_default_robustness_variant() -> None:
    """
    Validate robustness runs require a non-default variant label.
    """
    with pytest.raises(ValueError):
        ExperimentRun(
            run_id="run-1",
            candidate_id="cand-1",
            run_type=ExperimentRunType.ROBUSTNESS,
            variant_label="default",
            seed=7,
            status=ExperimentRunStatus.COMPLETED,
            metrics={"score": 0.7},
            started_at="2026-04-02T00:00:00+00:00",
            ended_at="2026-04-02T00:00:01+00:00",
        )


def test_robustness_assessment_summary_line_includes_risk_and_ratio() -> None:
    """
    Validate robustness assessment exposes a concise report summary.
    """
    assessment = RobustnessAssessment(
        assessment_id="assess-1",
        candidate_id="cand-1",
        variant_count=3,
        degradation_ratio=0.182,
        risk_level=RiskLevel.MEDIUM,
        risk_notes="Moderate degradation detected.",
        created_at="2026-04-02T00:00:00+00:00",
    )
    assert assessment.summary_line() == "medium risk across 3 variants (degradation=0.182)"


def test_robustness_runner_accepts_yaml_probe_matrix() -> None:
    """
    Validate the probe matrix loader accepts repository-style YAML syntax.
    """
    matrix_path = Path(".tmp-probe-matrix.yaml")
    matrix_path.write_text(
        "\n".join(
            [
                "# comment should be ignored",
                "gymnasium:",
                "  - variant_label: observation_dropout",
                "    seed: 11",
                "    overrides:",
                "      dropout_rate: 0.2",
                "isaacgym:",
                "  - variant_label: dynamics_shift",
                "    seed: 19",
                "    overrides:",
                "      friction_scale: 1.15",
            ]
        ),
        encoding="utf-8",
    )
    runner = RobustnessRunner(probe_matrix_path=matrix_path)
    assert runner._variants_for_backend("gymnasium")[0].variant_label == "observation_dropout"  # noqa: SLF001
    assert runner._variants_for_backend("isaacgym")[0].overrides == {"friction_scale": 1.15}  # noqa: SLF001
    matrix_path.unlink(missing_ok=True)


def test_retry_returns_on_eventual_success() -> None:
    """
    Validate retry wrapper returns once operation succeeds.
    """
    attempts = {"count": 0}

    def op() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    value = run_with_retry(op, policy=RetryPolicy(max_attempts=3, base_backoff_seconds=0))
    assert value == "ok"
    assert attempts["count"] == 3


def test_retry_raises_after_exhaustion() -> None:
    """
    Validate retry wrapper raises when attempts are exhausted.
    """
    with pytest.raises(RetryError):
        run_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("always fails")),
            policy=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
        )


def test_state_machine_allows_expected_transitions() -> None:
    """
    Validate expected transitions are marked valid.
    """
    assert can_transition(SessionLifecycleState.DRAFT, SessionLifecycleState.RUNNING)
    assert can_transition(SessionLifecycleState.RUNNING, SessionLifecycleState.PAUSED)


def test_state_machine_rejects_invalid_transitions() -> None:
    """
    Validate disallowed transitions raise explicit errors.
    """
    with pytest.raises(ValueError):
        ensure_transition(SessionLifecycleState.COMPLETED, SessionLifecycleState.RUNNING)


def test_session_repository_bootstraps_local_storage() -> None:
    """
    Validate repository initializes local SQLite and accepts session writes.
    """
    base_path = Path(".tmp-test-store")
    shutil.rmtree(base_path, ignore_errors=True)
    repo = SessionRepository(base_path)
    created = repo.create_session(
        SessionConfig(
            objective_text="stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=6,
            feedback_gate=FeedbackGate.NONE,
        )
    )
    stored = repo.get_session(created["session_id"])
    assert stored is not None
    assert stored["status"] == "running"
    shutil.rmtree(base_path, ignore_errors=True)
