"""
Summary: Backend adapter factory for environment-specific experiment routing.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.experiments.backends.base import EnvironmentBackendAdapter
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackendAdapter
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackendAdapter
from rewardlab.schemas.session_config import EnvironmentBackend


def resolve_backend_adapter(backend: EnvironmentBackend) -> EnvironmentBackendAdapter:
    """
    Build the adapter that matches the requested session backend.

    Args:
        backend: Target environment backend identifier.

    Returns:
        Backend adapter instance for the requested environment runtime.
    """
    if backend is EnvironmentBackend.GYMNASIUM:
        return GymnasiumBackendAdapter()
    if backend is EnvironmentBackend.ISAACGYM:
        return IsaacGymBackendAdapter()
    raise ValueError(f"unsupported backend: {backend}")
