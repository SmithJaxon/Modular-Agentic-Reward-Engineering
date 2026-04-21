"""Run bounded Isaac Gym worker smoke tests for Cartpole/Humanoid/AllegroHand."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def _text_tail(value: object, limit: int = 20) -> str:
    """Return a safe text tail from subprocess output that may be bytes."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return "\n".join(text.splitlines()[-limit:])


def _reward_source() -> str:
    return (
        "from __future__ import annotations\n\n"
        "def reward(env_reward: float = 0.0, **kwargs: object) -> float:\n"
        "    del kwargs\n"
        "    return float(env_reward)\n"
    )


def _request_payload(environment_id: str, seed: int) -> dict[str, object]:
    return {
        "execution_request": {
            "run_id": f"smoke-{environment_id.lower()}",
            "backend": "isaacgym",
            "environment_id": environment_id,
            "run_type": "performance",
            "execution_mode": "actual_backend",
            "variant_label": "smoke",
            "seed": seed,
            "entrypoint_name": "reward",
            "render_mode": None,
            "max_episode_steps": None,
        },
        "reward_program": {
            "candidate_id": f"smoke-{environment_id.lower()}-candidate",
            "source_text": _reward_source(),
            "entrypoint_name": "reward",
        },
        "policy_config": {
            "total_timesteps": 16,
            "checkpoint_count": 1,
            "evaluation_run_count": 1,
            "evaluation_episodes_per_checkpoint": 1,
            "n_envs": 1,
            "device": "cpu",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "evaluation_max_steps_floor": 8,
        },
    }


def run_one(environment_id: str, timeout_seconds: int, seed: int) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"rewardlab_smoke_{environment_id.lower()}_") as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(
            json.dumps(_request_payload(environment_id=environment_id, seed=seed)),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [
                    "python",
                    "-m",
                    "rewardlab.experiments.isaacgym_worker",
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                capture_output=True,
                text=True,
                timeout=max(timeout_seconds, 30),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.perf_counter() - started, 2)
            return {
                "environment_id": environment_id,
                "status": "fail",
                "reason": f"timeout_after_{timeout_seconds}s",
                "elapsed_seconds": elapsed,
                "worker_exit_code": None,
                "stdout_tail": _text_tail(exc.stdout),
                "stderr_tail": _text_tail(exc.stderr),
                "response": None,
            }
        elapsed = round(time.perf_counter() - started, 2)
        response_payload: dict[str, object] | None = None
        if response_path.exists():
            response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        status = "pass"
        reason = ""
        if completed.returncode != 0:
            status = "fail"
            reason = f"worker_exit={completed.returncode}"
        if response_payload is None:
            status = "fail"
            reason = "missing_response_payload"
        elif response_payload.get("status") != "ok":
            status = "fail"
            reason = str(response_payload.get("error", "worker_error"))
        return {
            "environment_id": environment_id,
            "status": status,
            "reason": reason,
            "elapsed_seconds": elapsed,
            "worker_exit_code": completed.returncode,
            "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
            "response": response_payload,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac Gym smoke matrix")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--env",
        action="append",
        choices=["Cartpole", "Humanoid", "AllegroHand"],
        help="Optional environment(s) to run; defaults to all.",
    )
    args = parser.parse_args()

    matrix = tuple(args.env) if args.env else ("Cartpole", "Humanoid", "AllegroHand")
    results = [run_one(env, timeout_seconds=args.timeout_seconds, seed=7) for env in matrix]
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
