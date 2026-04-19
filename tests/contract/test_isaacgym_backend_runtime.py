"""
Summary: Contract tests for Isaac Gym runtime readiness and task checks.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

import pytest

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend


class FakeIsaacGymEnvsModule:
    """Minimal IsaacGymEnvs module double with configurable version."""

    __version__ = "0.0-test"

    def make(self, **kwargs):  # noqa: ANN003
        """Return a sentinel environment payload."""

        return kwargs


def test_isaacgym_backend_reports_ready_with_injected_module() -> None:
    """Injected module and factory should permit a ready status."""

    backend = IsaacGymBackend(
        environment_factory=lambda **_: object(),
        isaacgymenvs_module=FakeIsaacGymEnvsModule(),
    )

    status = backend.get_runtime_status("Humanoid")

    assert status.ready is True
    assert status.backend.value == "isaacgym"
    assert status.detected_version == "0.0-test"


def test_isaacgym_backend_requires_non_blank_environment_id() -> None:
    """Blank task identifiers should produce a readiness failure."""

    backend = IsaacGymBackend(isaacgymenvs_module=FakeIsaacGymEnvsModule())

    status = backend.get_runtime_status("")

    assert status.ready is False
    assert "must not be blank" in status.status_reason


def test_isaacgym_backend_create_environment_raises_when_runtime_not_ready() -> None:
    """Environment creation should fail when runtime prerequisites are missing."""

    backend = IsaacGymBackend()

    with pytest.raises(RuntimeError, match="isaacgymenvs"):
        backend.create_environment("Humanoid")

