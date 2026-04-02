"""
Summary: Gymnasium backend adapter with lazy imports and test-friendly factories.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from rewardlab.experiments.backends.base import BackendAdapter, BackendEnvironment


class GymnasiumBackend(BackendAdapter):
    """Backend adapter for Gymnasium environments."""

    def __init__(
        self,
        environment_factory: Callable[..., BackendEnvironment] | None = None,
    ) -> None:
        """Store an optional factory override for local tests."""

        self._environment_factory = environment_factory

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
        try:
            import gymnasium as gym  # type: ignore[import-not-found]

            gym.spec(environment_id)
        except Exception:
            return False
        return True

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

        import gymnasium as gym  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return cast(BackendEnvironment, gym.make(environment_id, **kwargs))
