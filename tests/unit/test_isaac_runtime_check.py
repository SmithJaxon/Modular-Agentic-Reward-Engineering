"""Unit tests for Isaac runtime-check split-runtime worker probing."""

from __future__ import annotations

import json
import types

from rewardlab.cli import experiment_commands as commands
from rewardlab.schemas.agent_experiment import ExecutionIsaacConfig


def test_probe_isaac_worker_health_parses_success(monkeypatch) -> None:
    """Worker healthcheck JSON should be parsed and returned as ready payload."""

    completed = types.SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "status": "ok",
                "runtime_status": {"ready": True},
                "checks": {
                    "task_status": {
                        "Cartpole": {"ready": True},
                        "Humanoid": {"ready": True},
                        "AllegroHand": {"ready": True},
                    },
                    "available_tasks": ["Cartpole", "Humanoid", "AllegroHand"],
                    "config_dir": "/tmp/cfg",
                },
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(commands.subprocess, "run", lambda *args, **kwargs: completed)

    payload = commands._probe_isaac_worker_health(
        ["python", "worker.py"],
        task_ids=["Cartpole", "Humanoid", "AllegroHand"],
    )

    assert payload["status"] == "ok"
    assert payload["checks"]["task_status"]["Cartpole"]["ready"] is True


def test_probe_isaac_worker_health_handles_failure(monkeypatch) -> None:
    """Worker healthcheck command failure should return structured error payload."""

    completed = types.SimpleNamespace(
        returncode=2,
        stdout="",
        stderr="boom",
    )
    monkeypatch.setattr(commands.subprocess, "run", lambda *args, **kwargs: completed)

    payload = commands._probe_isaac_worker_health(
        ["python", "worker.py"],
        task_ids=["Cartpole"],
    )

    assert payload["status"] == "error"
    assert "failed with exit=2" in payload["error"]


def test_runtime_check_prefers_worker_task_status(monkeypatch) -> None:
    """runtime-check should use worker-side status in split-runtime mode."""

    class _Status:
        def model_dump(self, mode: str = "json"):  # noqa: ANN001
            del mode
            return {"backend": "isaacgym", "ready": False, "status_reason": "controller missing"}

    class _Backend:
        def __init__(self, cfg_dir_override=None):  # noqa: ANN001, D401
            del cfg_dir_override

        def get_runtime_status(self, _task_id):  # noqa: ANN001
            return _Status()

        def list_available_tasks(self):
            return []

        def resolve_config_dir(self):
            return None

    monkeypatch.setattr(commands, "IsaacGymBackend", _Backend)
    monkeypatch.setattr(commands, "_collect_isaac_import_status", lambda: {"torch_importable": False})
    monkeypatch.setattr(commands, "resolve_worker_command", lambda _cfg: ["python", "worker.py"])
    monkeypatch.setattr(
        commands,
        "_probe_isaac_worker_health",
        lambda _cmd, task_ids: {
            "status": "ok",
            "checks": {
                "task_status": {
                    task_id: {"backend": "isaacgym", "ready": True, "status_reason": "ok"}
                    for task_id in task_ids
                },
                "available_tasks": list(task_ids),
                "config_dir": "/worker/cfg",
            },
        },
    )

    payload = commands._isaac_runtime_check_payload(
        environment_ids=["Cartpole", "Humanoid", "AllegroHand"],
        isaac_config=ExecutionIsaacConfig(worker_command="python worker.py"),
    )

    assert payload["status"] == "ok"
    assert payload["checks"]["task_status"]["Cartpole"]["ready"] is True
    assert payload["checks"]["config_dir"] == "/worker/cfg"
