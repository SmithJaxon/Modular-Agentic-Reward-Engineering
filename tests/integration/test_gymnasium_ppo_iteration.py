"""
Summary: Integration tests for PPO-backed Gymnasium iteration with synthetic LLM output.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import shutil
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import pytest

from rewardlab.orchestrator.iteration_engine import IterationEngine
from rewardlab.orchestrator.session_service import SessionService
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.robustness_assessment import RiskLevel, RobustnessAssessment
from rewardlab.schemas.session_config import (
    EnvironmentBackend,
    FeedbackGate,
    SessionConfig,
)
from rewardlab.selection.risk_analyzer import RiskAnalysisResult


class _FakeOpenAIClient:
    """
    Return a deterministic reward program while capturing the prompt text.
    """

    def __init__(self) -> None:
        """
        Initialize prompt capture for assertions.
        """
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, *, max_output_tokens: int = 300) -> str:
        """
        Return a simple executable reward program for CartPole.
        """
        _ = max_output_tokens
        self.prompts.append(prompt)
        return """
```python
def compute_reward(observation, action, next_observation, env_reward, terminated, truncated, info):
    upright_bonus = 1.0 - abs(float(next_observation[2]))
    center_bonus = 1.0 - min(1.0, abs(float(next_observation[0])))
    reward = 0.8 * float(env_reward) + 0.15 * upright_bonus + 0.05 * center_bonus
    return reward, {
        "env_reward": float(env_reward),
        "upright_bonus": upright_bonus,
        "center_bonus": center_bonus,
    }
```
"""


class _StubRobustnessRunner:
    """
    Skip repeated PPO robustness retraining during the integration test.
    """

    def run(self, candidate_id: str, payload: object, primary_score: float) -> object:
        """
        Return a low-risk assessment with a small positive robustness bonus.
        """
        _ = payload
        analysis = RiskAnalysisResult(
            assessment=RobustnessAssessment(
                assessment_id="assess-test",
                candidate_id=candidate_id,
                variant_count=1,
                degradation_ratio=0.0,
                risk_level=RiskLevel.LOW,
                risk_notes="Low robustness risk for PPO integration test.",
                created_at="2026-04-06T00:00:00+00:00",
            ),
            robustness_bonus=0.05,
            tradeoff_rationale=None,
            minor_robustness_risk_accepted=False,
        )
        return type(
            "_Result",
            (),
            {
                "analysis": analysis,
                "assessment": analysis.assessment,
                "experiment_runs": [],
            },
        )()


@pytest.mark.integration
def test_session_step_can_run_ppo_with_llm_generated_reward() -> None:
    """
    Verify PPO-backed Gymnasium stepping can synthesize, train, and persist one reward candidate.
    """
    if find_spec("stable_baselines3") is None:
        pytest.skip("stable-baselines3 is not installed in this environment")

    root = Path(".tmp-gymnasium-ppo") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    fake_client = _FakeOpenAIClient()
    service = SessionService(
        repository=SessionRepository(root / "data"),
        iteration_engine=IterationEngine(openai_client=fake_client),
        robustness_runner=_StubRobustnessRunner(),
    )
    session = service.start_session(
        config=SessionConfig(
            objective_text="keep the pole upright and centered for as long as possible",
            environment_id="CartPole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=2,
            max_iterations=2,
            feedback_gate=FeedbackGate.NONE,
            metadata={
                "execution_mode": "ppo",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "ppo_total_timesteps": 64,
                "ppo_num_envs": 1,
                "ppo_n_steps": 64,
                "ppo_batch_size": 64,
                "ppo_learning_rate": 0.0003,
                "evaluation_episodes": 1,
                "reflection_episodes": 1,
                "reflection_interval_steps": 64,
                "train_seed": 7,
                "robustness_budget_scale": 0.5,
                "human_feedback_enabled": False,
                "peer_feedback_enabled": False,
            },
        ),
        baseline_reward_definition="Need a reward that keeps CartPole balanced.",
    )

    step = service.step_session(session["session_id"])
    candidate = service._repository.get_candidate(step["candidate_id"])  # noqa: SLF001
    reflection = service._repository.get_latest_reflection_for_candidate(  # noqa: SLF001
        step["candidate_id"]
    )
    artifact_dir = (
        root
        / "data"
        / "experiments"
        / "gymnasium"
        / session["session_id"]
        / "iter-000"
        / "default"
    )

    assert fake_client.prompts
    assert candidate is not None
    assert reflection is not None
    assert "def compute_reward" in candidate["reward_definition"]
    assert step["performance_summary"].startswith("gymnasium performance")
    assert "remaining_budget" in step
    assert "env_return:" in reflection["summary"]
    assert (artifact_dir / "policy_model.zip").exists()
    assert (artifact_dir / "summary.json").exists()
    assert (artifact_dir / "reflection.txt").exists()

    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.integration
def test_session_step_tracks_and_exhausts_adaptive_budget() -> None:
    """
    Verify PPO sessions can self-terminate once the adaptive budget is spent.
    """
    if find_spec("stable_baselines3") is None:
        pytest.skip("stable-baselines3 is not installed in this environment")

    root = Path(".tmp-gymnasium-budget") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    service = SessionService(
        repository=SessionRepository(root / "data"),
        iteration_engine=IterationEngine(openai_client=_FakeOpenAIClient()),
        robustness_runner=_StubRobustnessRunner(),
    )
    session = service.start_session(
        config=SessionConfig(
            objective_text="keep the pole upright and centered for as long as possible",
            environment_id="CartPole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=5,
            max_iterations=5,
            feedback_gate=FeedbackGate.NONE,
            metadata={
                "execution_mode": "ppo",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "budget_mode": "adaptive",
                "total_training_timesteps": 64,
                "total_evaluation_episodes": 2,
                "max_llm_calls": 1,
                "ppo_total_timesteps": 64,
                "ppo_num_envs": 1,
                "ppo_n_steps": 64,
                "ppo_batch_size": 64,
                "ppo_learning_rate": 0.0003,
                "evaluation_episodes": 1,
                "reflection_episodes": 1,
                "reflection_interval_steps": 64,
                "train_seed": 7,
                "robustness_budget_scale": 0.5,
                "human_feedback_enabled": False,
                "peer_feedback_enabled": False,
            },
        ),
        baseline_reward_definition="Need a reward that keeps CartPole balanced.",
    )

    step = service.step_session(session["session_id"])
    updated_session = service._repository.get_session(session["session_id"])  # noqa: SLF001

    assert updated_session is not None
    assert step["status"] == "completed"
    assert updated_session["stop_reason"] == "budget_cap"
    assert updated_session["metadata"]["spent_training_timesteps"] == 64
    assert updated_session["metadata"]["spent_evaluation_episodes"] == 1
    assert updated_session["metadata"]["llm_calls_used"] == 1
    assert step["remaining_budget"] == {
        "remaining_training_timesteps": 0,
        "remaining_evaluation_episodes": 1,
        "remaining_llm_calls": 0,
    }

    shutil.rmtree(root, ignore_errors=True)
