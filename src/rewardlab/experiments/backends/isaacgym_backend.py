"""
Summary: Isaac Gym backend adapter with lazy imports and test-friendly factories.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from rewardlab.experiments.backends.base import BackendAdapter, BackendEnvironment
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.utils.env import load_runtime_environment

EnvironmentFactory = Callable[..., BackendEnvironment]
EnvironmentValidator = Callable[[str], None]


class IsaacGymBackend(BackendAdapter):
    """Backend adapter for Isaac Gym-style environments."""

    def __init__(
        self,
        environment_factory: EnvironmentFactory | None = None,
        isaac_module: Any | None = None,
        environment_validator: EnvironmentValidator | None = None,
        environment_factory_entrypoint: str | None = None,
        environment_validator_entrypoint: str | None = None,
    ) -> None:
        """Store optional runtime and factory overrides for local tests and CLI use."""

        self._environment_factory = environment_factory
        self._isaac_module = isaac_module
        self._environment_validator = environment_validator
        self._environment_factory_entrypoint = environment_factory_entrypoint
        self._environment_validator_entrypoint = environment_validator_entrypoint

    @property
    def backend_name(self) -> str:
        """Return the stable backend identifier."""

        return "isaacgym"

    def supports(self, environment_id: str) -> bool:
        """Return whether the adapter can execute the requested environment."""

        return self.get_runtime_status(environment_id).ready

    def get_runtime_status(self, environment_id: str) -> BackendRuntimeStatus:
        """Return whether Isaac runtime prerequisites are satisfied for the environment."""

        if not environment_id:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason="environment_id must not be blank for Isaac execution",
                missing_prerequisites=["provide an approved Isaac environment id"],
            )

        isaac_module = self._load_isaacgym_module()
        if isaac_module is None:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason="isaacgym is not installed in the active worktree-local .venv",
                missing_prerequisites=[
                    "install approved Isaac runtime dependencies inside .venv"
                ],
            )

        environment_factory, factory_error = self._resolve_environment_factory()
        if factory_error is not None:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason=factory_error,
                missing_prerequisites=[
                    "set REWARDLAB_ISAAC_ENV_FACTORY to module.submodule:callable"
                ],
                detected_version=_detected_isaacgym_version(isaac_module),
            )
        if environment_factory is None:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason=(
                    "isaacgym is importable, but REWARDLAB_ISAAC_ENV_FACTORY is not "
                    "configured for actual RewardLab execution"
                ),
                missing_prerequisites=[
                    "set REWARDLAB_ISAAC_ENV_FACTORY to module.submodule:callable"
                ],
                detected_version=_detected_isaacgym_version(isaac_module),
            )

        environment_validator, validator_error = self._resolve_environment_validator()
        if validator_error is not None:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAACGYM,
                ready=False,
                status_reason=validator_error,
                missing_prerequisites=[
                    "fix REWARDLAB_ISAAC_ENV_VALIDATOR or unset it for factory-only validation"
                ],
                detected_version=_detected_isaacgym_version(isaac_module),
            )
        if environment_validator is not None:
            try:
                environment_validator(environment_id)
            except Exception as exc:
                return BackendRuntimeStatus(
                    backend=EnvironmentBackend.ISAACGYM,
                    ready=False,
                    status_reason=(
                        f"Isaac environment {environment_id!r} is not available: {exc}"
                    ),
                    missing_prerequisites=[
                        f"verify that the configured Isaac runtime supports {environment_id!r}"
                    ],
                    detected_version=_detected_isaacgym_version(isaac_module),
                )

        return BackendRuntimeStatus(
            backend=EnvironmentBackend.ISAACGYM,
            ready=True,
            status_reason=(
                "Isaac runtime and RewardLab environment factory are configured for "
                f"{environment_id!r}"
            ),
            detected_version=_detected_isaacgym_version(isaac_module),
        )

    def create_environment(
        self,
        environment_id: str,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BackendEnvironment:
        """Create an Isaac Gym environment or delegate to the injected factory."""

        if self._environment_factory is not None and self._isaac_module is None:
            return self._environment_factory(
                environment_id=environment_id,
                seed=seed,
                render_mode=render_mode,
            )

        runtime_status = self.get_runtime_status(environment_id)
        if not runtime_status.ready:
            raise RuntimeError(runtime_status.status_reason)

        environment_factory, factory_error = self._resolve_environment_factory()
        if environment_factory is None:
            message = factory_error or (
                "REWARDLAB_ISAAC_ENV_FACTORY must be configured for Isaac execution"
            )
            raise RuntimeError(message)

        try:
            return environment_factory(
                environment_id=environment_id,
                seed=seed,
                render_mode=render_mode,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Isaac environment {environment_id!r} failed to initialize: {exc}"
            ) from exc

    def _load_isaacgym_module(self) -> Any | None:
        """Return the injected or imported Isaac module when available."""

        if self._isaac_module is not None:
            return self._isaac_module
        try:
            return import_module("isaacgym")
        except Exception:
            return None

    def _resolve_environment_factory(self) -> tuple[EnvironmentFactory | None, str | None]:
        """Return the configured Isaac environment factory and any config error."""

        if self._environment_factory is not None:
            return self._environment_factory, None

        configured_entrypoint = self._environment_factory_entrypoint or self._runtime_environment(
            "REWARDLAB_ISAAC_ENV_FACTORY"
        )
        if configured_entrypoint is None:
            return None, None
        return _load_configured_callable(
            configured_entrypoint,
            env_name="REWARDLAB_ISAAC_ENV_FACTORY",
        )

    def _resolve_environment_validator(
        self,
    ) -> tuple[EnvironmentValidator | None, str | None]:
        """Return the configured Isaac environment validator and any config error."""

        if self._environment_validator is not None:
            return self._environment_validator, None

        configured_entrypoint = self._environment_validator_entrypoint or self._runtime_environment(
            "REWARDLAB_ISAAC_ENV_VALIDATOR"
        )
        if configured_entrypoint is None:
            return None, None
        return _load_configured_callable(
            configured_entrypoint,
            env_name="REWARDLAB_ISAAC_ENV_VALIDATOR",
        )

    def _runtime_environment(self, key: str) -> str | None:
        """Return one configured runtime setting from `.env` or process environment."""

        value = load_runtime_environment().get(key)
        if value is None or not value.strip():
            return None
        return value


def _detected_isaacgym_version(isaac_module: Any) -> str | None:
    """Return the best available version string for the loaded Isaac module."""

    version = getattr(isaac_module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version
    return None


def _load_configured_callable(
    entrypoint: str,
    *,
    env_name: str,
) -> tuple[Callable[..., Any] | None, str | None]:
    """Resolve a configured `module.submodule:callable` entrypoint string."""

    module_name, separator, attr_path = entrypoint.partition(":")
    if not separator or not module_name.strip() or not attr_path.strip():
        return None, f"{env_name} must use 'module.submodule:callable' syntax"

    try:
        module = import_module(module_name.strip())
    except Exception as exc:
        return None, f"{env_name} could not import {module_name!r}: {exc}"

    target: Any = module
    for attribute in attr_path.split("."):
        if not hasattr(target, attribute):
            return None, f"{env_name} could not resolve {attr_path!r} from {module_name!r}"
        target = getattr(target, attribute)

    if not callable(target):
        return None, f"{env_name} target {entrypoint!r} is not callable"
    return cast(Callable[..., Any], target), None
