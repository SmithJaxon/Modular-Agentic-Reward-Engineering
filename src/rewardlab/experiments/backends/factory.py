"""
Summary: Backend adapter resolution helpers for RewardLab experiment execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.experiments.backends.base import BackendAdapter
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.schemas.session_config import EnvironmentBackend


def resolve_backend(
    environment_backend: EnvironmentBackend | str,
    *,
    gymnasium_backend: BackendAdapter | None = None,
    isaacgym_backend: BackendAdapter | None = None,
) -> BackendAdapter:
    """Resolve a backend identifier to a concrete adapter instance."""

    backend_name = (
        environment_backend.value
        if isinstance(environment_backend, EnvironmentBackend)
        else environment_backend
    )
    if backend_name == EnvironmentBackend.GYMNASIUM.value:
        return gymnasium_backend or GymnasiumBackend()
    if backend_name == EnvironmentBackend.ISAAC_GYM.value:
        return isaacgym_backend or IsaacGymBackend()
    raise ValueError(f"unsupported environment backend: {backend_name!r}")
