"""
Summary: Runtime-gated Gymnasium smoke validation for supported local environments.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackendAdapter
from rewardlab.schemas.session_config import EnvironmentBackend


@pytest.mark.integration
def test_gymnasium_runtime_smoke_and_adapter_runtime_detection() -> None:
    """
    Verify Gymnasium can step a real environment and the adapter reports runtime availability.
    """
    if find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this environment")
    gymnasium = pytest.importorskip("gymnasium")

    env = gymnasium.make("CartPole-v1")
    total_reward = 0.0
    try:
        observation, info = env.reset(seed=7)
        assert observation is not None
        assert isinstance(info, dict)

        for _ in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            assert observation is not None
            assert isinstance(info, dict)
            if terminated or truncated:
                observation, info = env.reset(seed=7)
                assert observation is not None
                assert isinstance(info, dict)
    finally:
        env.close()

    output = GymnasiumBackendAdapter().run_performance(
        ExperimentInput(
            session_id="runtime-gymnasium",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            reward_definition="reward = stability_bonus + smooth_control_bonus",
            iteration_index=0,
            objective_text="maximize stability",
        )
    )

    assert total_reward >= 0.0
    assert output.metrics["runtime_available"] is True
