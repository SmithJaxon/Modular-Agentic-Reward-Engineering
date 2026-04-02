"""
Summary: Contract tests for Gymnasium and Isaac Gym backend adapters.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import pytest

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackendAdapter
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackendAdapter
from rewardlab.schemas.session_config import EnvironmentBackend


@pytest.mark.contract
@pytest.mark.parametrize(
    ("adapter", "backend"),
    [
        (GymnasiumBackendAdapter(), EnvironmentBackend.GYMNASIUM),
        (IsaacGymBackendAdapter(), EnvironmentBackend.ISAACGYM),
    ],
)
def test_backend_adapters_return_deterministic_outputs(
    adapter: object,
    backend: EnvironmentBackend,
) -> None:
    """
    Verify each backend adapter returns stable, backend-tagged outputs.
    """
    payload = ExperimentInput(
        session_id="session-1",
        environment_id="cartpole-v1",
        environment_backend=backend,
        reward_definition="reward = shaped_progress_bonus",
        iteration_index=2,
        objective_text="maximize stability",
    )

    first = adapter.run_performance(payload)
    second = adapter.run_performance(payload)
    reflection = adapter.run_reflection(payload)

    assert first == second
    assert first.metrics["backend"] == backend.value
    assert first.metrics["variant_label"] == "default"
    assert "runtime_available" in first.metrics
    assert first.summary.startswith(f"{backend.value} performance")
    assert reflection.metrics["backend"] == backend.value
    assert reflection.summary.startswith(f"{backend.value} reflection")


@pytest.mark.contract
def test_backend_adapters_preserve_non_default_variant_metadata() -> None:
    """
    Verify probe variants are surfaced without requiring real backend installs.
    """
    payload = ExperimentInput(
        session_id="session-2",
        environment_id="isaac-ant-v0",
        environment_backend=EnvironmentBackend.ISAACGYM,
        reward_definition="reward = exploit_bonus + survival_bonus",
        iteration_index=1,
        objective_text="maximize forward progress",
        variant_label="observation_dropout",
        seed=17,
        overrides={"dropout_rate": 0.25},
    )

    output = IsaacGymBackendAdapter().run_performance(payload)

    assert output.metrics["variant_label"] == "observation_dropout"
    assert output.metrics["seed"] == 17
    assert output.metrics["overrides"] == {"dropout_rate": 0.25}
