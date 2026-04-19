"""Debug helper for running one IsaacGym worker request in-container."""

from __future__ import annotations

import json
import os
import subprocess


def main() -> None:
    source = (
        "from __future__ import annotations\n\n"
        "def reward(env_reward: float = 0.0, **kwargs: object) -> float:\n"
        "    del kwargs\n"
        "    return float(env_reward)\n"
    )
    request = {
        "execution_request": {
            "run_id": "debug-cartpole-run",
            "backend": "isaacgym",
            "environment_id": "Cartpole",
            "run_type": "performance",
            "execution_mode": "actual_backend",
            "variant_label": "default",
            "seed": 7,
            "entrypoint_name": "reward",
            "render_mode": None,
            "max_episode_steps": None,
        },
        "reward_program": {
            "candidate_id": "debug-candidate",
            "source_text": source,
            "entrypoint_name": "reward",
        },
        "policy_config": {
            "total_timesteps": 2000,
            "checkpoint_count": 1,
            "evaluation_run_count": 1,
            "evaluation_episodes_per_checkpoint": 1,
            "n_envs": 4,
            "device": "cpu",
            "learning_rate": 3e-4,
            "gamma": 0.99,
        },
    }
    request_path = "/tmp/isaac_worker_req.json"
    response_path = "/tmp/isaac_worker_resp.json"
    with open(request_path, "w", encoding="utf-8") as handle:
        json.dump(request, handle)
    completed = subprocess.run(
        [
            "python",
            "-m",
            "rewardlab.experiments.isaacgym_worker",
            "--request",
            request_path,
            "--response",
            response_path,
        ],
        check=False,
        timeout=600,
    )
    print("worker_rc", completed.returncode)
    print("response_exists", os.path.exists(response_path))
    if os.path.exists(response_path):
        with open(response_path, "r", encoding="utf-8") as handle:
            print(handle.read())


if __name__ == "__main__":
    main()
