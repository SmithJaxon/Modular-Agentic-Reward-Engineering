"""
Summary: Contract tests for Isaac Gym runtime readiness and actionable failures.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend


class FakeIsaacModule:
    """Minimal Isaac module double exposing a version string for readiness checks."""

    __version__ = "preview-0.1"


class FakeIsaacEnvironment:
    """Minimal environment double for create-environment contract checks."""

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Return a trivial observation payload."""

        return [0.0, 0.0], {"seed": seed}

    def step(
        self,
        action: list[float],
    ) -> tuple[list[float], float, bool, bool, dict[str, list[float]]]:
        """Terminate immediately after one action."""

        return [0.0, 0.0], 0.0, True, False, {"action": action}

    def close(self) -> None:
        """Close the fake environment handle."""


def test_isaacgym_backend_reports_missing_runtime_when_module_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Isaac runtime packages should produce an actionable readiness failure."""

    monkeypatch.delenv("REWARDLAB_ISAAC_ENV_FACTORY", raising=False)

    backend = IsaacGymBackend()
    status = backend.get_runtime_status("Isaac-Cartpole-v0")

    assert status.ready is False
    assert status.backend.value == "isaacgym"
    assert ".venv" in status.status_reason
    assert status.missing_prerequisites


def test_isaacgym_backend_reports_factory_prerequisite_when_runtime_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed runtime still needs a configured RewardLab environment factory."""

    monkeypatch.delenv("REWARDLAB_ISAAC_ENV_FACTORY", raising=False)

    backend = IsaacGymBackend(isaac_module=FakeIsaacModule())
    status = backend.get_runtime_status("Isaac-Cartpole-v0")

    assert status.ready is False
    assert "REWARDLAB_ISAAC_ENV_FACTORY" in status.status_reason
    assert any("REWARDLAB_ISAAC_ENV_FACTORY" in item for item in status.missing_prerequisites)


def test_isaacgym_backend_reports_actionable_error_for_unknown_environment() -> None:
    """An unsupported Isaac environment id should fail before environment creation."""

    backend = IsaacGymBackend(
        environment_factory=lambda **_: FakeIsaacEnvironment(),
        isaac_module=FakeIsaacModule(),
        environment_validator=_validator_for({"Isaac-Cartpole-v0"}),
    )

    status = backend.get_runtime_status("MissingIsaacEnv-v0")

    assert status.ready is False
    assert "MissingIsaacEnv-v0" in status.status_reason
    assert status.missing_prerequisites

    with pytest.raises(RuntimeError, match="MissingIsaacEnv-v0"):
        backend.create_environment("MissingIsaacEnv-v0")


def test_isaacgym_backend_reports_ready_status_when_runtime_and_factory_are_configured() -> None:
    """A configured runtime plus factory should produce a ready status."""

    backend = IsaacGymBackend(
        environment_factory=lambda **_: FakeIsaacEnvironment(),
        isaac_module=FakeIsaacModule(),
        environment_validator=_validator_for({"Isaac-Cartpole-v0"}),
    )

    status = backend.get_runtime_status("Isaac-Cartpole-v0")

    assert status.ready is True
    assert status.backend.value == "isaacgym"
    assert status.detected_version == "preview-0.1"


def _validator_for(environment_ids: set[str]) -> Callable[[str], None]:
    """Return a validator that only accepts the provided environment ids."""

    def validate(environment_id: str) -> None:
        """Raise when the requested environment is not supported."""

        if environment_id not in environment_ids:
            raise RuntimeError(f"Unsupported Isaac environment: {environment_id}")

    return validate
