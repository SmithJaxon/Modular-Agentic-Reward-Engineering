#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mare.reward_candidate import RewardCandidateLoader


RewardFn = Callable[[Any, Any, Dict[str, Any]], float]


@dataclass
class TrainArtifacts:
    train_request_path: Path
    result_path: Path
    checkpoint_path: Optional[Path] = None


class CartPoleEnv:
    """Small self-contained CartPole implementation for local experiments."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    max_episode_steps = 500

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.state = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        x, x_dot, theta, theta_dot = [float(value) for value in self.state]
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        info = {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "terminated": terminated,
            "truncated": truncated,
            "steps": self.steps,
            "default_reward": 1.0,
        }
        return self.state.copy(), 1.0, terminated, truncated, info


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(observations)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local CartPole trainer and placeholder IsaacGym entrypoint")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reward-candidate", type=Path, default=None)
    parser.add_argument("--reward-entrypoint", default="compute_reward")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    request_payload: Dict[str, Any] = {
        "environment": args.environment,
        "task_name": args.task_name,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "device": args.device,
        "reward_candidate": str(args.reward_candidate) if args.reward_candidate is not None else None,
        "reward_entrypoint": args.reward_entrypoint,
    }
    train_request_path = args.run_dir / "train_request.json"
    train_request_path.write_text(
        json.dumps(request_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    start = time.time()
    reward_fn = load_reward_function(args.reward_candidate, args.reward_entrypoint)

    if args.environment != "CartPole":
        result = placeholder_remote_only_result(args, train_request_path)
        write_result(args.run_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    device, device_warning = select_device(args.device)
    set_global_seed(args.seed)

    if args.algorithm.upper() == "PPO":
        training = train_cartpole_ppo(args.seed, args.train_steps, device, reward_fn)
    elif args.algorithm.upper() == "A2C":
        training = train_cartpole_a2c(args.seed, args.train_steps, device, reward_fn)
    elif args.algorithm.upper() == "DDQN":
        training = train_cartpole_ddqn(args.seed, args.train_steps, device, reward_fn)
    else:
        raise SystemExit("Unsupported local algorithm: {0}".format(args.algorithm))

    evaluation_score = evaluate_cartpole(
        algorithm=args.algorithm.upper(),
        model=training["model"],
        seed=args.seed + 10_000,
        episodes=args.eval_episodes,
        device=device,
        reward_fn=reward_fn,
    )

    checkpoint_path = args.run_dir / "policy.pt"
    save_checkpoint(args.algorithm.upper(), training["model"], checkpoint_path)
    result = {
        "status": "completed",
        "metrics": {
            "seed": float(args.seed),
            "train_steps": float(args.train_steps),
            "eval_episodes": float(args.eval_episodes),
            "training_score": round(float(training["training_score"]), 6),
            "evaluation_score": round(float(evaluation_score), 6),
            "episodes_completed": float(training["episodes_completed"]),
            "wall_time_sec": round(time.time() - start, 6),
        },
        "warnings": [device_warning] if device_warning is not None else [],
        "artifacts": [
            {"name": "train_request", "path": str(train_request_path)},
            {"name": "result", "path": str(args.run_dir / "result.json")},
            {"name": "policy", "path": str(checkpoint_path)},
        ],
        "notes": "Local CartPole {0} run completed with a self-contained trainer on {1}.".format(
            args.algorithm.upper(),
            device.type,
        ),
        "reward_candidate": {
            "path": str(args.reward_candidate) if args.reward_candidate is not None else None,
            "entrypoint": args.reward_entrypoint,
        },
    }
    write_result(args.run_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def placeholder_remote_only_result(args: argparse.Namespace, train_request_path: Path) -> Dict[str, Any]:
    return {
        "status": "remote_runtime_required",
        "metrics": {
            "seed": float(args.seed),
            "train_steps": float(args.train_steps),
            "eval_episodes": float(args.eval_episodes),
        },
        "warnings": [
            "Local runtime is implemented only for CartPole. Isaac Gym-backed environments still require the GPU VM."
        ],
        "artifacts": [
            {"name": "train_request", "path": str(train_request_path)},
            {"name": "result", "path": str(args.run_dir / "result.json")},
        ],
        "notes": "Execution plan recorded, but no local runtime exists for {0}.".format(args.environment),
        "reward_candidate": {
            "path": str(args.reward_candidate) if args.reward_candidate is not None else None,
            "entrypoint": args.reward_entrypoint,
        },
    }


def write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reward_function(path: Optional[Path], entrypoint: str) -> Optional[RewardFn]:
    if path is None:
        return None
    loader = RewardCandidateLoader()
    return loader.load(path, entrypoint=entrypoint).entrypoint


def select_device(device_name: str) -> Tuple[torch.device, Optional[str]]:
    requested = device_name.lower()
    if requested == "cpu":
        return torch.device("cpu"), None
    if requested in {"auto", "cuda"}:
        cuda_ready, detail = probe_cuda_runtime()
        if cuda_ready:
            return torch.device("cuda"), None
        if requested == "cuda":
            return (
                torch.device("cpu"),
                "Requested CUDA, but the local Torch/CUDA stack could not run on this machine; fell back to CPU. {0}".format(detail),
            )
        return (
            torch.device("cpu"),
            "Auto device selection chose CPU because CUDA was unavailable or unsupported. {0}".format(detail),
        )
    return torch.device("cpu"), "Unknown device '{0}' requested; fell back to CPU.".format(device_name)


def probe_cuda_runtime() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned false."
    try:
        probe = torch.zeros(1, device="cuda")
        _ = probe + 1
    except Exception as exc:
        return False, "{0}: {1}".format(type(exc).__name__, exc)
    return True, "CUDA runtime probe succeeded."


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reward_value(reward_fn: Optional[RewardFn], observation: np.ndarray, action: int, info: Dict[str, Any]) -> float:
    if reward_fn is None:
        return float(info.get("default_reward", 1.0))
    return float(reward_fn(observation.tolist(), action, dict(info)))


def train_cartpole_ppo(seed: int, train_steps: int, device: torch.device, reward_fn: Optional[RewardFn]) -> Dict[str, Any]:
    env = CartPoleEnv(seed)
    model = ActorCriticNet(obs_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    rollout_steps = min(256, max(32, train_steps))
    update_epochs = 4
    minibatch_size = 64

    episode_returns: List[float] = []
    current_return = 0.0
    observation = env.reset()
    steps = 0

    while steps < train_steps:
        storage_obs: List[np.ndarray] = []
        storage_actions: List[int] = []
        storage_log_probs: List[float] = []
        storage_rewards: List[float] = []
        storage_values: List[float] = []
        storage_dones: List[bool] = []

        for _ in range(min(rollout_steps, train_steps - steps)):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                distribution = Categorical(logits=logits)
                action = int(distribution.sample().item())
                log_prob = float(distribution.log_prob(torch.tensor(action, device=device)).item())
                value_item = float(value.item())

            next_observation, _, terminated, truncated, info = env.step(action)
            reward = reward_value(reward_fn, next_observation, action, info)

            storage_obs.append(observation.copy())
            storage_actions.append(action)
            storage_log_probs.append(log_prob)
            storage_rewards.append(reward)
            storage_values.append(value_item)
            storage_dones.append(terminated or truncated)

            current_return += reward
            observation = next_observation
            steps += 1

            if terminated or truncated:
                episode_returns.append(current_return)
                current_return = 0.0
                observation = env.reset()

        with torch.no_grad():
            next_obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = model(next_obs_tensor)
            bootstrap_value = float(next_value.item())

        returns, advantages = generalized_advantage_estimate(
            rewards=storage_rewards,
            values=storage_values,
            dones=storage_dones,
            next_value=bootstrap_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        obs_batch = torch.tensor(np.asarray(storage_obs), dtype=torch.float32, device=device)
        actions_batch = torch.tensor(storage_actions, dtype=torch.long, device=device)
        old_log_probs_batch = torch.tensor(storage_log_probs, dtype=torch.float32, device=device)
        returns_batch = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_batch = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        batch_size = obs_batch.shape[0]
        for _ in range(update_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                logits, values = model(obs_batch[indices])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions_batch[indices])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs_batch[indices])
                unclipped = ratio * advantages_batch[indices]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_batch[indices]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns_batch[indices])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    return {
        "model": model,
        "training_score": trailing_average(episode_returns),
        "episodes_completed": len(episode_returns),
    }


def train_cartpole_a2c(seed: int, train_steps: int, device: torch.device, reward_fn: Optional[RewardFn]) -> Dict[str, Any]:
    env = CartPoleEnv(seed)
    model = ActorCriticNet(obs_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    gamma = 0.99
    rollout_steps = min(128, max(32, train_steps))
    episode_returns: List[float] = []
    current_return = 0.0
    observation = env.reset()
    steps = 0

    while steps < train_steps:
        storage_obs: List[np.ndarray] = []
        storage_actions: List[int] = []
        storage_rewards: List[float] = []
        storage_values: List[float] = []
        storage_dones: List[bool] = []

        for _ in range(min(rollout_steps, train_steps - steps)):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(obs_tensor)
            distribution = Categorical(logits=logits)
            action = int(distribution.sample().item())
            next_observation, _, terminated, truncated, info = env.step(action)
            reward = reward_value(reward_fn, next_observation, action, info)

            storage_obs.append(observation.copy())
            storage_actions.append(action)
            storage_rewards.append(reward)
            storage_values.append(float(value.item()))
            storage_dones.append(terminated or truncated)

            current_return += reward
            observation = next_observation
            steps += 1

            if terminated or truncated:
                episode_returns.append(current_return)
                current_return = 0.0
                observation = env.reset()

        with torch.no_grad():
            next_obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = model(next_obs_tensor)

        returns = discounted_returns(storage_rewards, storage_dones, float(next_value.item()), gamma)
        advantages = [ret - value for ret, value in zip(returns, storage_values)]

        obs_batch = torch.tensor(np.asarray(storage_obs), dtype=torch.float32, device=device)
        actions_batch = torch.tensor(storage_actions, dtype=torch.long, device=device)
        returns_batch = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_batch = torch.tensor(advantages, dtype=torch.float32, device=device)

        logits, values = model(obs_batch)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions_batch)
        entropy = distribution.entropy().mean()
        policy_loss = -(log_probs * advantages_batch.detach()).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns_batch)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    return {
        "model": model,
        "training_score": trailing_average(episode_returns),
        "episodes_completed": len(episode_returns),
    }


def train_cartpole_ddqn(seed: int, train_steps: int, device: torch.device, reward_fn: Optional[RewardFn]) -> Dict[str, Any]:
    env = CartPoleEnv(seed)
    online = QNetwork(obs_dim=4, action_dim=2).to(device)
    target = QNetwork(obs_dim=4, action_dim=2).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    gamma = 0.99
    batch_size = 64
    min_replay = 128
    update_target_every = 100
    replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=10_000)

    episode_returns: List[float] = []
    current_return = 0.0
    observation = env.reset()

    for step in range(train_steps):
        epsilon = max(0.05, 1.0 - (0.95 * (step / max(1, train_steps))))
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = online(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        next_observation, _, terminated, truncated, info = env.step(action)
        reward = reward_value(reward_fn, next_observation, action, info)
        done = terminated or truncated
        replay.append((observation.copy(), action, reward, next_observation.copy(), done))
        current_return += reward
        observation = next_observation

        if done:
            episode_returns.append(current_return)
            current_return = 0.0
            observation = env.reset()

        if len(replay) >= min_replay:
            batch = random.sample(replay, batch_size)
            obs_batch = torch.tensor(np.asarray([item[0] for item in batch]), dtype=torch.float32, device=device)
            actions_batch = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            rewards_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            next_obs_batch = torch.tensor(np.asarray([item[3] for item in batch]), dtype=torch.float32, device=device)
            dones_batch = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)

            q_values = online(obs_batch).gather(1, actions_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(online(next_obs_batch), dim=1)
                next_q = target(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards_batch + gamma * (1.0 - dones_batch) * next_q
            loss = torch.nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            optimizer.step()

        if (step + 1) % update_target_every == 0:
            target.load_state_dict(online.state_dict())

    return {
        "model": online,
        "training_score": trailing_average(episode_returns),
        "episodes_completed": len(episode_returns),
    }


def evaluate_cartpole(
    algorithm: str,
    model: nn.Module,
    seed: int,
    episodes: int,
    device: torch.device,
    reward_fn: Optional[RewardFn],
) -> float:
    env = CartPoleEnv(seed)
    returns: List[float] = []
    for episode_index in range(max(1, episodes)):
        observation = env.reset()
        episode_return = 0.0
        for _ in range(env.max_episode_steps):
            action = greedy_action(algorithm, model, observation, device)
            next_observation, _, terminated, truncated, info = env.step(action)
            reward = reward_value(reward_fn, next_observation, action, info)
            episode_return += reward
            observation = next_observation
            if terminated or truncated:
                break
        returns.append(episode_return)
        env._rng = np.random.default_rng(seed + episode_index + 1)
    return float(sum(returns) / len(returns))


def greedy_action(algorithm: str, model: nn.Module, observation: np.ndarray, device: torch.device) -> int:
    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if algorithm in {"PPO", "A2C"}:
            logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())
        q_values = model(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def generalized_advantage_estimate(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages = [0.0 for _ in rewards]
    returns = [0.0 for _ in rewards]
    gae = 0.0
    future_value = next_value
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * future_value * mask - values[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
        returns[index] = gae + values[index]
        future_value = values[index]
    return returns, advantages


def discounted_returns(
    rewards: Sequence[float],
    dones: Sequence[bool],
    next_value: float,
    gamma: float,
) -> List[float]:
    returns = [0.0 for _ in rewards]
    running = next_value
    for index in reversed(range(len(rewards))):
        if dones[index]:
            running = 0.0
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def trailing_average(values: Sequence[float], window: int = 10) -> float:
    if not values:
        return 0.0
    tail = list(values[-window:])
    return float(sum(tail) / len(tail))


def save_checkpoint(algorithm: str, model: nn.Module, path: Path) -> None:
    payload = {
        "algorithm": algorithm,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    raise SystemExit(main())
