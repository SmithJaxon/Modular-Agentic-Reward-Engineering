"""
Summary: Unit tests for Gymnasium reward-program normalization and reflection formatting.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import numpy as np
import pytest

from rewardlab.experiments.gymnasium_runtime import (
    EvaluationSnapshot,
    RewardProgram,
    build_reflection_summary,
)


def test_reward_program_accepts_fenced_source_and_normalizes_outputs() -> None:
    """
    Verify fenced reward-program source is compiled and returns normalized scalar outputs.
    """
    program = RewardProgram(
        """
        ```python
        def compute_reward(
            observation, action, next_observation, env_reward, terminated, truncated, info
        ):
            upright_bonus = 1.0 - abs(float(next_observation[2]))
            reward = float(env_reward) + 0.25 * upright_bonus
            return reward, {"env_reward": float(env_reward), "upright_bonus": upright_bonus}
        ```
        """
    )

    result = program.evaluate(
        observation=np.zeros(4, dtype=np.float32),
        action=np.zeros(1, dtype=np.float32),
        next_observation=np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32),
        env_reward=1.0,
        terminated=False,
        truncated=False,
        info={},
    )

    assert result.reward == pytest.approx(1.225)
    assert result.components["env_reward"] == pytest.approx(1.0)
    assert result.components["upright_bonus"] == pytest.approx(0.9)


def test_reflection_summary_includes_checkpoint_series_and_component_names() -> None:
    """
    Verify PPO reflection summaries surface checkpoint trends and final metrics.
    """
    summary = build_reflection_summary(
        environment_id="CartPole-v1",
        checkpoints=[
            EvaluationSnapshot(
                timesteps=64,
                env_return_mean=10.0,
                shaped_return_mean=11.0,
                episode_length_mean=10.0,
                component_means={"upright_bonus": 0.8},
            ),
            EvaluationSnapshot(
                timesteps=128,
                env_return_mean=15.0,
                shaped_return_mean=16.0,
                episode_length_mean=15.0,
                component_means={"upright_bonus": 0.9},
            ),
        ],
        final_snapshot=EvaluationSnapshot(
            timesteps=128,
            env_return_mean=18.0,
            shaped_return_mean=19.0,
            episode_length_mean=18.0,
            component_means={"upright_bonus": 0.95},
        ),
        reflection_interval_steps=64,
    )

    assert "Environment: CartPole-v1" in summary
    assert "env_return:" in summary
    assert "upright_bonus:" in summary
    assert "Final evaluation snapshot:" in summary
