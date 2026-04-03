"""
Summary: Contract tests for Gymnasium runtime readiness and actionable failures.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import pytest

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend


class FakeGymModule:
    """Minimal Gymnasium module double with controllable environment registration."""

    __version__ = "0.29.1"

    def __init__(self, registered_ids: set[str]) -> None:
        """Store the environment ids that should resolve successfully."""

        self.registered_ids = registered_ids

    def spec(self, environment_id: str) -> object:
        """Return a dummy spec object when the environment is registered."""

        if environment_id not in self.registered_ids:
            raise RuntimeError(f"No registered env with id: {environment_id}")
        return object()


class FakeEnvironment:
    """Minimal environment double for create-environment contract checks."""

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, int | None]]:
        """Return a trivial observation payload."""

        return 0, {"seed": seed}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
        """Terminate immediately after one action."""

        return 0, 0.0, True, False, {"action": action}

    def close(self) -> None:
        """Close the fake environment handle."""


def test_gymnasium_backend_reports_ready_status_for_registered_environment() -> None:
    """A registered environment should produce a ready runtime status."""

    backend = GymnasiumBackend(
        environment_factory=lambda **_: FakeEnvironment(),
        gym_module=FakeGymModule({"CartPole-v1"}),
    )

    status = backend.get_runtime_status("CartPole-v1")

    assert status.ready is True
    assert status.backend.value == "gymnasium"
    assert status.detected_version == "0.29.1"


def test_gymnasium_backend_reports_actionable_error_for_unknown_environment() -> None:
    """An unknown environment id should produce a clear readiness failure."""

    backend = GymnasiumBackend(gym_module=FakeGymModule({"CartPole-v1"}))

    status = backend.get_runtime_status("MissingEnv-v0")

    assert status.ready is False
    assert "MissingEnv-v0" in status.status_reason
    assert status.missing_prerequisites

    with pytest.raises(RuntimeError, match="MissingEnv-v0"):
        backend.create_environment("MissingEnv-v0")
