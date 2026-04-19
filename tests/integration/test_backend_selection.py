"""
Summary: Integration test for environment backend resolution and routing.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import pytest

from rewardlab.experiments.backends.factory import resolve_backend
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.schemas.session_config import EnvironmentBackend


def test_backend_resolution_routes_to_requested_adapter() -> None:
    """Backend resolution should honor the requested environment backend value."""

    gymnasium_backend = GymnasiumBackend(environment_factory=lambda **_: None)

    assert (
        resolve_backend(
            EnvironmentBackend.GYMNASIUM,
            gymnasium_backend=gymnasium_backend,
        )
        is gymnasium_backend
    )


def test_backend_resolution_rejects_unknown_backend() -> None:
    """Backend resolution should reject unsupported backend identifiers."""

    with pytest.raises(ValueError):
        resolve_backend("unsupported")


def test_backend_resolution_supports_isaacgym_adapter() -> None:
    """Backend resolution should route Isaac Gym requests to provided adapters."""

    isaac_backend = IsaacGymBackend(environment_factory=lambda **_: None)

    assert (
        resolve_backend(
            EnvironmentBackend.ISAAC_GYM,
            isaacgym_backend=isaac_backend,
        )
        is isaac_backend
    )
