"""
Summary: Integration tests for robustness probes, risk analysis, and policy tradeoffs.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.factory import resolve_backend_adapter
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.orchestrator.iteration_engine import IterationResult
from rewardlab.orchestrator.session_service import SessionService
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.schemas.session_report import RiskLevel


class _StubIterationEngine:
    """
    Emit deterministic candidate results to exercise policy tradeoff selection.
    """

    def __init__(self) -> None:
        """
        Seed fixed iteration outputs for two-step integration coverage.
        """
        self._results = [
            IterationResult(
                reward_definition="reward = stability_bonus + smooth_control_bonus",
                change_summary="safe candidate",
                score=0.61,
                performance_metrics={},
                performance_summary="gymnasium performance score=0.610",
                reflection_summary="safe candidate reflection",
                proposed_changes=["keep stability weighting"],
                confidence=0.72,
            ),
            IterationResult(
                reward_definition="reward = speed_bonus + fragile_progress_bonus",
                change_summary="higher performance with narrow tuning",
                score=0.72,
                performance_metrics={},
                performance_summary="gymnasium performance score=0.720",
                reflection_summary="medium-risk candidate reflection",
                proposed_changes=["accept some fragility for throughput"],
                confidence=0.81,
            ),
        ]

    def run_iteration(
        self,
        session: dict[str, object],
        iteration_index: int,
        baseline_reward_definition: str,
    ) -> IterationResult:
        """
        Return the pre-seeded result for the requested iteration.
        """
        _ = session
        _ = baseline_reward_definition
        return self._results[iteration_index]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("backend", "environment_id"),
    [
        (EnvironmentBackend.GYMNASIUM, "cartpole-v1"),
        (EnvironmentBackend.ISAACGYM, "isaac-ant-v0"),
    ],
)
def test_probe_matrix_flags_reward_hacking_risk(
    backend: EnvironmentBackend,
    environment_id: str,
) -> None:
    """
    Verify configured robustness probes classify exploit-prone rewards as high risk.
    """
    payload = ExperimentInput(
        session_id="session-risk",
        environment_id=environment_id,
        environment_backend=backend,
        reward_definition="reward = exploit_bonus + hack_progress_bonus",
        iteration_index=1,
        objective_text="maximize stable task completion",
    )
    primary = resolve_backend_adapter(backend).run_performance(payload)
    runner = RobustnessRunner()

    result = runner.run(candidate_id="cand-risk", payload=payload, primary_score=primary.score)

    assert len(result.experiment_runs) >= 2
    assert {run.variant_label for run in result.experiment_runs} == {
        "observation_dropout",
        "dynamics_shift",
        "reward_delay",
    }
    assert result.assessment.risk_level is RiskLevel.HIGH
    assert "reward-hacking risk" in result.assessment.risk_notes


@pytest.mark.integration
def test_policy_records_rationale_when_minor_robustness_risk_is_accepted() -> None:
    """
    Verify a medium-risk but higher-value candidate carries explicit tradeoff rationale.
    """
    root = Path(".tmp-reward-hack-probes") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports"

    service = SessionService(
        repository=SessionRepository(root / "data"),
        iteration_engine=_StubIterationEngine(),
    )
    session = service.start_session(
        config=SessionConfig(
            objective_text="maximize stability",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=2,
            max_iterations=2,
            feedback_gate=FeedbackGate.NONE,
        ),
        baseline_reward_definition="reward = base_bonus",
    )

    service.step_session(session["session_id"])
    second_step = service.step_session(session["session_id"])
    report = service.report_session(session["session_id"], report_dir=report_dir)
    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert payload["best_candidate"]["candidate_id"] == second_step["best_candidate_id"]
    assert payload["best_candidate"]["minor_robustness_risk_accepted"] is True
    assert "minor robustness risk" in payload["best_candidate"]["selection_summary"].lower()
    shutil.rmtree(root, ignore_errors=True)
