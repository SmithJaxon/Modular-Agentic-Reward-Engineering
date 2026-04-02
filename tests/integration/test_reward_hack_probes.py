"""
Summary: Integration test for reward-hacking probe execution and risk analysis.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pathlib import Path

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.selection.policy import CandidateSelectionPolicy


class FakeEnvironment:
    """Environment double that returns a single configured reward value."""

    def __init__(self, reward: float) -> None:
        """Store a single-step reward for deterministic probe runs."""

        self.reward = reward

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, int | None]]:
        """Return a trivial initial observation payload."""

        return 0, {"seed": seed}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
        """Terminate after one step with the configured reward."""

        return 1, self.reward, True, False, {"action": action}

    def close(self) -> None:
        """Close the fake environment handle."""


def test_reward_hack_probe_matrix_flags_risk_and_penalizes_selection(tmp_path: Path) -> None:
    """Probe execution should produce a high-risk assessment when variants collapse."""

    reward_by_environment = {
        "cartpole-v1-primary": 10.0,
        "cartpole-v1-noisy": 4.0,
        "cartpole-v1-shifted": 3.0,
    }
    backend = GymnasiumBackend(
        environment_factory=lambda environment_id, **_: FakeEnvironment(
            reward_by_environment[environment_id]
        )
    )
    runner = RobustnessRunner(
        probe_matrix_path=Path("tools/reward_hack_probes/probe_matrix.yaml"),
        gymnasium_backend=backend,
    )
    risky_candidate = RewardCandidate(
        candidate_id="candidate-risky",
        session_id="session-001",
        iteration_index=2,
        reward_definition="def reward(state): return 10.0",
        change_summary="Risky but strong candidate.",
        aggregate_score=10.0,
    )
    safe_candidate = RewardCandidate(
        candidate_id="candidate-safe",
        session_id="session-001",
        iteration_index=1,
        reward_definition="def reward(state): return 9.0",
        change_summary="Slightly weaker but stable candidate.",
        aggregate_score=9.2,
    )

    runs, assessment = runner.run_candidate_probes(
        candidate=risky_candidate,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        environment_id="cartpole-v1",
        policy=lambda _: 0,
    )
    selected = CandidateSelectionPolicy().select_best_candidate(
        [risky_candidate, safe_candidate],
        assessments={risky_candidate.candidate_id: assessment},
    )

    assert len(runs) >= 3
    assert assessment.risk_level.value == "high"
    assert assessment.variant_count == len(runs)
    assert selected.candidate_id == safe_candidate.candidate_id
