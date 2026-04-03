"""
Summary: Gymnasium backend adapter with lazy imports and test-friendly factories.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from rewardlab.experiments.backends.base import BackendAdapter, BackendEnvironment
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend


class GymnasiumBackend(BackendAdapter):
    """Backend adapter for Gymnasium environments."""

    def __init__(
        self,
        environment_factory: Callable[..., BackendEnvironment] | None = None,
        gym_module: Any | None = None,
    ) -> None:
        """Store optional environment and module overrides for local tests."""

        self._environment_factory = environment_factory
        self._gym_module = gym_module

    @property
    def backend_name(self) -> str:
        """Return the stable backend identifier."""

        return "gymnasium"

    def supports(self, environment_id: str) -> bool:
        """Return whether the adapter can execute the requested environment."""

        if not environment_id:
            return False
        if self._environment_factory is not None:
            return True

        gym = self._load_gymnasium_module()
        if gym is None:
            return False
        try:
            gym.spec(environment_id)
        except Exception:
            return False
        return True

    def get_runtime_status(self, environment_id: str) -> BackendRuntimeStatus:
        """Return whether Gymnasium is ready for the requested environment id."""

        if not environment_id:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.GYMNASIUM,
                ready=False,
                status_reason="environment_id must not be blank for Gymnasium execution",
                missing_prerequisites=["provide a registered Gymnasium environment id"],
            )

        gym = self._load_gymnasium_module()
        if gym is None:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.GYMNASIUM,
                ready=False,
                status_reason="gymnasium is not installed in the active worktree-local .venv",
                missing_prerequisites=[
                    "install approved RewardLab Gymnasium dependencies inside .venv"
                ],
            )

        try:
            gym.spec(environment_id)
        except Exception as exc:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.GYMNASIUM,
                ready=False,
                status_reason=(
                    f"Gymnasium environment {environment_id!r} is not available: {exc}"
                ),
                missing_prerequisites=[
                    f"verify that Gymnasium registers environment {environment_id!r}"
                ],
                detected_version=_detected_gymnasium_version(gym),
            )

        return BackendRuntimeStatus(
            backend=EnvironmentBackend.GYMNASIUM,
            ready=True,
            status_reason=(
                f"Gymnasium import and environment resolution succeeded for {environment_id!r}"
            ),
            detected_version=_detected_gymnasium_version(gym),
        )

    def create_environment(
        self,
        environment_id: str,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BackendEnvironment:
        """Create a Gymnasium environment or delegate to the injected factory."""

        if self._environment_factory is not None:
            return self._environment_factory(
                environment_id=environment_id,
                seed=seed,
                render_mode=render_mode,
            )

        runtime_status = self.get_runtime_status(environment_id)
        if not runtime_status.ready:
            raise RuntimeError(runtime_status.status_reason)

        gym = self._require_gymnasium_module()
        kwargs: dict[str, Any] = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return cast(BackendEnvironment, gym.make(environment_id, **kwargs))

    def _load_gymnasium_module(self) -> Any | None:
        """Return the injected or imported Gymnasium module when available."""

        if self._gym_module is not None:
            return self._gym_module
        try:
            import gymnasium as gym  # type: ignore[import-not-found]
        except Exception:
            return None
        return gym

    def _require_gymnasium_module(self) -> Any:
        """Return the Gymnasium module or raise if it is unavailable."""

        gym = self._load_gymnasium_module()
        if gym is None:
            raise RuntimeError("gymnasium is not installed in the active worktree-local .venv")
        return gym


def _detected_gymnasium_version(gym_module: Any) -> str | None:
    """Return the best available version string for the loaded Gymnasium module."""

    version = getattr(gym_module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version
    return None
