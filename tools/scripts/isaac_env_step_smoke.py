"""Smoke-check Isaac Gym task init/reset/single-step for key environments."""

from __future__ import annotations

import json
import time

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend


def _zero_action(environment, torch):
    action_space = getattr(environment, "action_space", None)
    n = getattr(action_space, "n", None)
    if isinstance(n, int) and n > 0:
        return torch.zeros((1,), dtype=torch.long, device="cpu")
    shape = getattr(action_space, "shape", None)
    action_dim = int(shape[-1]) if shape else 1
    return torch.zeros((1, action_dim), dtype=torch.float32, device="cpu")


def run_one(environment_id: str) -> dict[str, object]:
    import torch

    backend = IsaacGymBackend()
    started = time.perf_counter()
    env = backend.create_task_environment(
        environment_id=environment_id,
        seed=7,
        num_envs=1,
        sim_device="cpu",
        rl_device="cpu",
        headless=True,
    )
    try:
        env.reset()
        action = _zero_action(env, torch)
        env.step(action)
    finally:
        if hasattr(env, "close"):
            env.close()
        else:
            gym_handle = getattr(env, "gym", None)
            sim_handle = getattr(env, "sim", None)
            if gym_handle is not None and sim_handle is not None and hasattr(gym_handle, "destroy_sim"):
                gym_handle.destroy_sim(sim_handle)
    return {
        "environment_id": environment_id,
        "status": "pass",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def main() -> int:
    matrix = ("Cartpole", "Humanoid", "AllegroHand")
    results = []
    for env in matrix:
        try:
            results.append(run_one(env))
        except Exception as exc:
            results.append(
                {
                    "environment_id": env,
                    "status": "fail",
                    "reason": str(exc),
                }
            )
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
