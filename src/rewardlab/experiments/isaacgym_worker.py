"""
Summary: Isolated worker entrypoint for one Isaac Gym run.
Created: 2026-04-19
Last Updated: 2026-04-19
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from rewardlab.experiments.execution_service import ExecutionRequest
from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.experiments.isaacgym_runner import (
    IsaacGymPolicyConfig,
    _execute_in_process,
)
from rewardlab.experiments.reward_program import load_reward_program
from rewardlab.schemas.experiment_run import ExecutionMode, RunType
from rewardlab.schemas.session_config import EnvironmentBackend


def main() -> int:
    """Run one isolated Isaac execution and write structured JSON response."""

    parser = argparse.ArgumentParser(description="Isolated Isaac Gym run worker")
    parser.add_argument("--request", help="Path to JSON request payload")
    parser.add_argument("--response", help="Path to JSON response payload")
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run runtime checks only and emit structured JSON.",
    )
    args = parser.parse_args()

    if args.healthcheck:
        payload = _healthcheck_payload()
        serialized = json.dumps(payload)
        if args.response:
            Path(args.response).write_text(serialized, encoding="utf-8")
        else:
            print(serialized)
        return 0 if payload.get("status") == "ok" else 1

    if not args.request or not args.response:
        parser.error("--request and --response are required unless --healthcheck is used")

    request_path = Path(args.request)
    response_path = Path(args.response)
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        execution_request = _request_from_payload(payload.get("execution_request", {}))
        reward_payload = payload.get("reward_program", {})
        reward_program = load_reward_program(
            candidate_id=str(reward_payload.get("candidate_id", "worker-candidate")),
            source_text=str(reward_payload.get("source_text", "")),
            entrypoint_name=str(reward_payload.get("entrypoint_name", "reward")),
            runtime_compat_profile="isaacgym",
        )
        if reward_program.validation_status.value != "valid":
            response_path.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "error": reward_program.validation_error or "invalid reward program",
                        "runtime_status": None,
                    }
                ),
                encoding="utf-8",
            )
            return 0
        policy_config = IsaacGymPolicyConfig(**payload.get("policy_config", {}))
        backend_config = payload.get("backend_config", {})
        cfg_dir_override = None
        if isinstance(backend_config, dict):
            raw_cfg_dir = backend_config.get("cfg_dir_override")
            if isinstance(raw_cfg_dir, str) and raw_cfg_dir.strip():
                cfg_dir_override = raw_cfg_dir
        outcome = _execute_in_process(
            execution_request=execution_request,
            reward_program=reward_program,
            backend=IsaacGymBackend(cfg_dir_override=cfg_dir_override),
            policy_config=policy_config,
            close_environment=False,
        )
        response_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "outcome": {
                        "metrics": outcome.metrics,
                        "event_trace": outcome.event_trace,
                        "runtime_status": (
                            None if outcome.runtime_status is None else outcome.runtime_status.model_dump()
                        ),
                        "manifest_metadata": outcome.manifest_metadata,
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive worker guard
        response = {
            "status": "error",
            "error": f"{exc}",
            "runtime_status": None,
            "traceback": traceback.format_exc(),
        }
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return 0


def _request_from_payload(payload: dict[str, Any]) -> ExecutionRequest:
    """Build typed execution request from JSON payload."""

    return ExecutionRequest(
        run_id=str(payload.get("run_id", "")),
        backend=EnvironmentBackend(str(payload.get("backend", "isaacgym"))),
        environment_id=str(payload.get("environment_id", "")),
        run_type=RunType(str(payload.get("run_type", "performance"))),
        execution_mode=ExecutionMode(str(payload.get("execution_mode", "actual_backend"))),
        variant_label=str(payload.get("variant_label", "default")),
        seed=payload.get("seed"),
        entrypoint_name=str(payload.get("entrypoint_name", "reward")),
        render_mode=payload.get("render_mode"),
        max_episode_steps=payload.get("max_episode_steps"),
    )


def _healthcheck_payload() -> dict[str, Any]:
    """Return worker-runtime readiness payload for split-runtime probing."""

    backend = IsaacGymBackend()
    task_ids = ("Cartpole", "Humanoid", "AllegroHand")
    task_status = {
        task_id: backend.get_runtime_status(task_id).model_dump(mode="json")
        for task_id in task_ids
    }
    all_ready = all(bool(status.get("ready")) for status in task_status.values())
    checks = {
        "python_executable": sys.executable,
        "task_status": task_status,
        "available_tasks": sorted(backend.list_available_tasks()),
        "config_dir": backend.resolve_config_dir(),
    }
    return {
        "status": "ok" if all_ready else "error",
        "runtime_status": {
            "backend": EnvironmentBackend.ISAAC_GYM.value,
            "ready": all_ready,
            "status_reason": (
                "Isaac worker healthcheck passed"
                if all_ready
                else "Isaac worker healthcheck failed for one or more task targets"
            ),
            "missing_prerequisites": (
                []
                if all_ready
                else ["isaacgym worker runtime prerequisites"]
            ),
            "detected_version": None,
        },
        "checks": checks,
    }


if __name__ == "__main__":
    raise SystemExit(main())
