"""
Summary: Isaac Gym backend adapter with runtime readiness checks.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from rewardlab.experiments.backends.base import BackendAdapter, BackendEnvironment
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend


class IsaacGymBackend(BackendAdapter):
    """Backend adapter for Isaac Gym environments via IsaacGymEnvs."""

    def __init__(
        self,
        environment_factory: Callable[..., BackendEnvironment] | None = None,
        isaacgymenvs_module: Any | None = None,
        cfg_dir_override: str | None = None,
    ) -> None:
        """Store optional environment and module overrides for tests."""

        self._environment_factory = environment_factory
        self._isaacgymenvs_module = isaacgymenvs_module
        self._last_import_error: str | None = None
        self._cfg_dir_override = cfg_dir_override

    @property
    def backend_name(self) -> str:
        """Return the stable backend identifier."""

        return EnvironmentBackend.ISAAC_GYM.value

    def supports(self, environment_id: str) -> bool:
        """Return whether the adapter can execute the requested environment."""

        if not environment_id:
            return False
        if self._environment_factory is not None:
            return True
        status = self.get_runtime_status(environment_id)
        return status.ready

    def get_runtime_status(self, environment_id: str) -> BackendRuntimeStatus:
        """Return whether Isaac Gym is ready for the requested environment id."""

        if not environment_id:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAAC_GYM,
                ready=False,
                status_reason="environment_id must not be blank for Isaac Gym execution",
                missing_prerequisites=["provide a valid IsaacGymEnvs task id"],
            )

        isaacgymenvs = self._load_isaacgymenvs_module()
        if isaacgymenvs is None:
            detail = (
                f" (import error: {self._last_import_error})"
                if self._last_import_error
                else ""
            )
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAAC_GYM,
                ready=False,
                status_reason=(
                    "isaacgymenvs is not installed in the active .venv, "
                    f"or NVIDIA Isaac Gym bindings are unavailable{detail}"
                ),
                missing_prerequisites=[
                    "install approved IsaacGymEnvs dependencies in the worktree-local .venv",
                    "install NVIDIA Isaac Gym python package in the same .venv",
                ],
            )

        available_tasks = _resolve_available_tasks(isaacgymenvs)
        if available_tasks and environment_id not in available_tasks:
            return BackendRuntimeStatus(
                backend=EnvironmentBackend.ISAAC_GYM,
                ready=False,
                status_reason=(
                    f"IsaacGymEnvs task {environment_id!r} is not available in this runtime"
                ),
                missing_prerequisites=[
                    f"use one of the registered tasks: {', '.join(sorted(available_tasks))}"
                ],
                detected_version=_detected_isaacgymenvs_version(isaacgymenvs),
            )

        return BackendRuntimeStatus(
            backend=EnvironmentBackend.ISAAC_GYM,
            ready=True,
            status_reason=(
                f"Isaac Gym runtime is ready for task {environment_id!r}"
            ),
            detected_version=_detected_isaacgymenvs_version(isaacgymenvs),
        )

    def create_environment(
        self,
        environment_id: str,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BackendEnvironment:
        """Create a single-environment IsaacGymEnvs task instance."""

        del render_mode
        return self.create_task_environment(
            environment_id=environment_id,
            seed=seed,
            num_envs=1,
        )

    def create_task_environment(
        self,
        *,
        environment_id: str,
        seed: int | None = None,
        num_envs: int = 1,
        sim_device: str = "cuda:0",
        rl_device: str = "cuda:0",
        graphics_device_id: int = 0,
        headless: bool = True,
    ) -> BackendEnvironment:
        """Create an IsaacGymEnvs task with explicit vector-environment controls."""

        if self._environment_factory is not None:
            return self._environment_factory(
                environment_id=environment_id,
                seed=seed,
                num_envs=num_envs,
                sim_device=sim_device,
                rl_device=rl_device,
                graphics_device_id=graphics_device_id,
                headless=headless,
            )

        runtime_status = self.get_runtime_status(environment_id)
        if not runtime_status.ready:
            raise RuntimeError(runtime_status.status_reason)

        isaacgymenvs = self._require_isaacgymenvs_module()
        _ensure_torch_compile_compat_for_isaac()
        resolved_seed = seed if seed is not None else 0
        task_runtime = _resolve_task_runtime_profile(
            environment_id,
            sim_device=sim_device,
            rl_device=rl_device,
        )
        cfg = _compose_task_cfg(
            isaacgymenvs_module=isaacgymenvs,
            task=environment_id,
            num_envs=max(num_envs, 1),
            sim_device=sim_device,
            rl_device=rl_device,
            headless=headless,
            force_render=False,
            pipeline=task_runtime.pipeline,
            cfg_dir_override=self._cfg_dir_override,
        )
        environment = isaacgymenvs.make(
            seed=resolved_seed,
            task=environment_id,
            num_envs=max(num_envs, 1),
            sim_device=sim_device,
            rl_device=rl_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            force_render=False,
            cfg=cfg,
        )
        return cast(BackendEnvironment, environment)

    def _load_isaacgymenvs_module(self) -> Any | None:
        """Return the injected or imported IsaacGymEnvs module when available."""

        if self._isaacgymenvs_module is not None:
            return self._isaacgymenvs_module
        try:
            import isaacgym  # noqa: F401  # type: ignore[import-not-found]
            import isaacgymenvs  # type: ignore[import-not-found]
        except Exception as exc:
            self._last_import_error = f"{type(exc).__name__}: {exc}"
            return None
        self._last_import_error = None
        return isaacgymenvs

    def _require_isaacgymenvs_module(self) -> Any:
        """Return the IsaacGymEnvs module or raise if unavailable."""

        module = self._load_isaacgymenvs_module()
        if module is None:
            raise RuntimeError(
                "isaacgymenvs or NVIDIA Isaac Gym is not installed in the active .venv"
            )
        return module

    def list_available_tasks(self) -> set[str]:
        """Return registered IsaacGymEnvs tasks discovered in current runtime."""

        module = self._load_isaacgymenvs_module()
        if module is None:
            return set()
        return _resolve_available_tasks(module)

    def resolve_config_dir(self) -> str | None:
        """Resolve active IsaacGymEnvs config directory path."""

        module = self._load_isaacgymenvs_module()
        if module is None:
            return None
        return _resolve_isaac_cfg_dir(
            isaacgymenvs_module=module,
            cfg_dir_override=self._cfg_dir_override,
        )


def _resolve_available_tasks(isaacgymenvs_module: Any) -> set[str]:
    """Return the discovered set of registered IsaacGymEnvs task names."""

    try:
        from isaacgymenvs.tasks import isaacgym_task_map  # type: ignore[import-not-found]
    except Exception:
        return set()
    if isinstance(isaacgym_task_map, dict):
        return {str(key) for key in isaacgym_task_map}
    return set()


def _detected_isaacgymenvs_version(module: Any) -> str | None:
    """Return the best available version string for the loaded module."""

    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version
    return None


def _ensure_torch_compile_compat_for_isaac() -> None:
    """Backfill torch.compile in Isaac Gym runtime when using Torch 1.x."""

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return
    if hasattr(torch, "compile"):
        return

    def _compile_noop(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return fn

    torch.compile = _compile_noop  # type: ignore[attr-defined]


class _IsaacTaskRuntimeProfile:
    """Runtime profile for one IsaacGymEnvs task."""

    def __init__(self, *, pipeline: str) -> None:
        self.pipeline = pipeline


def _uses_cuda_device(device: str) -> bool:
    """Return whether the Isaac device string targets CUDA execution."""

    return device.strip().lower().startswith("cuda")


def _resolve_task_runtime_profile(
    environment_id: str,
    *,
    sim_device: str,
    rl_device: str,
) -> _IsaacTaskRuntimeProfile:
    """Return stable runtime settings for task families."""

    del environment_id
    pipeline = (
        "gpu"
        if _uses_cuda_device(sim_device) or _uses_cuda_device(rl_device)
        else "cpu"
    )
    return _IsaacTaskRuntimeProfile(pipeline=pipeline)


def _compose_task_cfg(
    *,
    isaacgymenvs_module: Any,
    task: str,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    headless: bool,
    force_render: bool,
    pipeline: str,
    cfg_dir_override: str | None,
) -> Any | None:
    """Compose an IsaacGymEnvs Hydra config for deterministic task runtime settings."""

    try:
        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.hydra_config import HydraConfig
    except Exception:
        return None

    if HydraConfig.initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    overrides = [
        f"task={task}",
        f"num_envs={num_envs}",
        f"sim_device={sim_device}",
        f"rl_device={rl_device}",
        f"pipeline={pipeline}",
        f"headless={'true' if headless else 'false'}",
        f"force_render={'true' if force_render else 'false'}",
    ]
    config_dir = _resolve_isaac_cfg_dir(
        isaacgymenvs_module=isaacgymenvs_module,
        cfg_dir_override=cfg_dir_override,
    )
    if config_dir is None:
        return None
    with initialize_config_dir(config_dir=config_dir):
        return compose(config_name="config", overrides=overrides)


def _resolve_isaac_cfg_dir(
    *,
    isaacgymenvs_module: Any,
    cfg_dir_override: str | None,
) -> str | None:
    """Resolve IsaacGymEnvs hydra config directory from override/env/module path."""

    candidate_override = (
        cfg_dir_override
        or os.getenv("REWARDLAB_ISAAC_CFG_DIR", "").strip()
        or None
    )
    if candidate_override:
        resolved = Path(candidate_override).expanduser().resolve()
        if resolved.exists():
            return str(resolved)

    module_file = getattr(isaacgymenvs_module, "__file__", None)
    if isinstance(module_file, str) and module_file:
        module_root = Path(module_file).resolve().parent
        cfg_dir = module_root / "cfg"
        if cfg_dir.exists():
            return str(cfg_dir)
    return None
