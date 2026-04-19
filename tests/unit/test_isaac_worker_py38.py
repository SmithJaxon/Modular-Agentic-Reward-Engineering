"""
Summary: Regression tests for the standalone Isaac Python 3.8 worker.
Created: 2026-04-19
Last Updated: 2026-04-19
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

WORKER_PATH = Path(__file__).resolve().parents[2] / "tools" / "scripts" / "isaac_worker_py38.py"


@pytest.fixture()
def worker_module():
    """Load the standalone Isaac worker module from disk."""

    spec = importlib.util.spec_from_file_location("isaac_worker_py38", WORKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict[str, Any]:
    """Build the minimal execution payload required by the worker."""

    return {
        "execution_request": {
            "environment_id": "Cartpole",
            "seed": 13,
        },
        "reward_program": {
            "source_text": "def reward(**kwargs): return 0.0",
            "entrypoint_name": "reward",
        },
        "policy_config": {
            "n_envs": 4,
            "device": "auto",
        },
        "backend_config": {},
    }


class _AbortExecutionError(RuntimeError):
    """Stop the worker after environment creation so tests can inspect make() kwargs."""


class _DummyEnv:
    """Minimal environment with a close method for worker cleanup."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_execute_uses_cfg_when_make_accepts_cfg(monkeypatch, worker_module) -> None:
    """The worker should pass composed Hydra cfg to runtimes that accept cfg."""

    calls: list[dict[str, Any]] = []

    class _IsaacGymEnvs:
        @staticmethod
        def make(
            seed: int,
            task: str,
            num_envs: int,
            sim_device: str,
            rl_device: str,
            headless: bool,
            force_render: bool,
            virtual_screen_capture: bool,
            cfg: dict[str, Any],
        ) -> _DummyEnv:
            calls.append(
                {
                    "seed": seed,
                    "task": task,
                    "num_envs": num_envs,
                    "sim_device": sim_device,
                    "rl_device": rl_device,
                    "headless": headless,
                    "force_render": force_render,
                    "virtual_screen_capture": virtual_screen_capture,
                    "cfg": cfg,
                }
            )
            return _DummyEnv()

    monkeypatch.setattr(
        worker_module,
        "_load_isaac_runtime",
        lambda cfg_dir_override=None: (
            object(),
            _IsaacGymEnvs,
            "C:/rewardlab/isaac-cfg",
            ["Cartpole"],
        ),
    )
    monkeypatch.setattr(worker_module, "_compose_task_cfg", lambda **kwargs: {"task": "Cartpole"})
    monkeypatch.setattr(
        worker_module,
        "_load_reward_callable",
        lambda **kwargs: (_ for _ in ()).throw(_AbortExecutionError("stop after make")),
    )

    with pytest.raises(_AbortExecutionError, match="stop after make"):
        worker_module._execute(_payload())

    assert len(calls) == 1
    assert calls[0]["cfg"] == {"task": "Cartpole"}
    assert "cfg_dir" not in calls[0]


def test_execute_falls_back_to_cfg_dir_when_make_rejects_cfg(monkeypatch, worker_module) -> None:
    """The worker should retry with cfg_dir for older make() signatures."""

    calls: list[dict[str, Any]] = []

    class _IsaacGymEnvs:
        @staticmethod
        def make(
            seed: int,
            task: str,
            num_envs: int,
            sim_device: str,
            rl_device: str,
            headless: bool,
            force_render: bool,
            virtual_screen_capture: bool,
            cfg_dir: str,
        ) -> _DummyEnv:
            calls.append(
                {
                    "seed": seed,
                    "task": task,
                    "num_envs": num_envs,
                    "sim_device": sim_device,
                    "rl_device": rl_device,
                    "headless": headless,
                    "force_render": force_render,
                    "virtual_screen_capture": virtual_screen_capture,
                    "cfg_dir": cfg_dir,
                }
            )
            return _DummyEnv()

    monkeypatch.setattr(
        worker_module,
        "_load_isaac_runtime",
        lambda cfg_dir_override=None: (
            object(),
            _IsaacGymEnvs,
            "C:/rewardlab/isaac-cfg",
            ["Cartpole"],
        ),
    )
    monkeypatch.setattr(worker_module, "_compose_task_cfg", lambda **kwargs: {"task": "Cartpole"})
    monkeypatch.setattr(
        worker_module,
        "_load_reward_callable",
        lambda **kwargs: (_ for _ in ()).throw(_AbortExecutionError("stop after make")),
    )

    with pytest.raises(_AbortExecutionError, match="stop after make"):
        worker_module._execute(_payload())

    assert len(calls) == 1
    assert calls[0]["cfg_dir"] == "C:/rewardlab/isaac-cfg"
    assert "cfg" not in calls[0]
