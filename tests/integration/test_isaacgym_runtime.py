"""
Summary: Runtime-gated Isaac Gym smoke validation for supported local environments.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import pytest

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackendAdapter
from rewardlab.schemas.session_config import EnvironmentBackend


@pytest.mark.integration
def test_isaacgym_runtime_smoke_and_adapter_runtime_detection() -> None:
    """
    Verify Isaac Gym can be imported on a supported machine and the adapter reports availability.
    """
    try:
        from isaacgym import gymapi
    except Exception as exc:  # pragma: no cover - environment-gated runtime skip.
        pytest.skip(f"isaacgym runtime import unavailable: {exc}")

    gym = gymapi.acquire_gym()
    output = IsaacGymBackendAdapter().run_performance(
        ExperimentInput(
            session_id="runtime-isaacgym",
            environment_id="isaac-ant-v0",
            environment_backend=EnvironmentBackend.ISAACGYM,
            reward_definition="reward = speed_bonus + stability_bonus",
            iteration_index=0,
            objective_text="maximize forward progress",
        )
    )

    assert gym is not None
    assert output.metrics["runtime_available"] is True
