#!/usr/bin/env python3
"""Standalone Isaac Gym worker for Python 3.8 split-runtime execution."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_json_dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _runtime_status(ready: bool, reason: str, missing: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "backend": "isaacgym",
        "ready": bool(ready),
        "status_reason": str(reason),
        "missing_prerequisites": list(missing or []),
        "detected_version": None,
    }


def _ensure_torch_compile_compat() -> None:
    import torch

    if not hasattr(torch, "compile"):
        torch.compile = lambda fn=None, *args, **kwargs: fn  # type: ignore[attr-defined]


def _resolve_cfg_dir(isaacgymenvs_module: Any, cfg_dir_override: Optional[str]) -> str:
    explicit = (
        (cfg_dir_override or "").strip()
        or os.getenv("REWARDLAB_ISAAC_CFG_DIR", "").strip()
        or ""
    )
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if not explicit_path.exists():
            raise RuntimeError(
                "IsaacGymEnvs cfg directory override does not exist: %s" % explicit_path
            )
        return str(explicit_path)

    module_file = getattr(isaacgymenvs_module, "__file__", None)
    if module_file:
        module_root = Path(module_file).resolve().parent
        cfg_dir = module_root / "cfg"
        if cfg_dir.exists():
            return str(cfg_dir)
    raise RuntimeError(
        "Could not resolve IsaacGymEnvs cfg directory. Set REWARDLAB_ISAAC_CFG_DIR "
        "or backend_config.cfg_dir_override."
    )


def _load_isaac_runtime(cfg_dir_override: Optional[str] = None) -> Tuple[Any, Any, str, List[str]]:
    import isaacgym  # noqa: F401
    import torch
    import isaacgymenvs
    from isaacgymenvs.tasks import isaacgym_task_map

    _ensure_torch_compile_compat()
    cfg_dir = _resolve_cfg_dir(isaacgymenvs_module=isaacgymenvs, cfg_dir_override=cfg_dir_override)
    tasks = sorted(str(key) for key in isaacgym_task_map.keys())
    return torch, isaacgymenvs, cfg_dir, tasks


def _healthcheck_payload() -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "python_executable": os.path.realpath(os.sys.executable),
    }
    missing: List[str] = []
    try:
        torch, _, cfg_dir, tasks = _load_isaac_runtime()
        checks["isaacgym_importable"] = True
        checks["isaacgymenvs_importable"] = True
        checks["config_dir"] = cfg_dir
        checks["available_tasks"] = tasks
        checks["torch_importable"] = True
        checks["torch_version"] = getattr(torch, "__version__", None)
        checks["cuda_available"] = bool(torch.cuda.is_available())
        checks["cuda_device_count"] = int(torch.cuda.device_count())
        task_status: Dict[str, Any] = {}
        for task_id in ("Cartpole", "Humanoid", "AllegroHand"):
            task_status[task_id] = _runtime_status(
                ready=task_id in tasks,
                reason=(
                    "Isaac Gym runtime is ready for task '%s'" % task_id
                    if task_id in tasks
                    else "Task '%s' is not registered in isaacgym_task_map" % task_id
                ),
                missing=[] if task_id in tasks else ["isaacgymenvs task registry"],
            )
        checks["task_status"] = task_status
    except Exception as exc:
        try:
            import torch

            checks["torch_importable"] = True
            checks["torch_version"] = getattr(torch, "__version__", None)
            checks["cuda_available"] = bool(torch.cuda.is_available())
            checks["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception as torch_exc:
            checks["torch_importable"] = False
            checks["torch_error"] = "%s: %s" % (type(torch_exc).__name__, torch_exc)
            checks["cuda_available"] = False
            checks["cuda_device_count"] = 0
        checks["isaacgym_importable"] = False
        checks["isaacgymenvs_importable"] = False
        checks["isaac_runtime_error"] = "%s: %s" % (type(exc).__name__, exc)
        if not checks.get("torch_importable") and "torch" not in missing:
            missing.append("torch")
        if "isaacgym" not in missing:
            missing.append("isaacgym")
        if "isaacgymenvs" not in missing:
            missing.append("isaacgymenvs")

    ready = len(missing) == 0 and bool(checks.get("task_status"))
    return {
        "status": "ok" if ready else "error",
        "runtime_status": _runtime_status(
            ready=ready,
            reason=(
                "Isaac worker healthcheck passed"
                if ready
                else "Isaac worker healthcheck failed: missing prerequisites"
            ),
            missing=missing,
        ),
        "checks": checks,
    }


def _resolve_device(configured: str, torch: Any) -> str:
    normalized = (configured or "auto").strip().lower()
    if normalized == "auto":
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("execution.ppo.device requests CUDA but CUDA is unavailable")
    return configured


def _seed_for_index(base_seed: Optional[int], offset: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)


def _as_tensor(value: Any, torch: Any, device: str) -> Any:
    if hasattr(value, "to") and hasattr(value, "dtype"):
        return value.to(device).float()
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _extract_observation(raw_observation: Any, torch: Any, device: str) -> Any:
    if isinstance(raw_observation, dict):
        candidate = raw_observation.get("obs")
        if candidate is None:
            raise RuntimeError("Isaac observation dictionary is missing 'obs'")
        tensor = _as_tensor(candidate, torch, device)
    else:
        tensor = _as_tensor(raw_observation, torch, device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    if isinstance(value, dict):
        return {key: _to_python_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    return value


def _normalize_info_list(raw_info: Any, env_count: int) -> List[Dict[str, Any]]:
    if isinstance(raw_info, list):
        normalized = [item if isinstance(item, dict) else {} for item in raw_info]
        if len(normalized) < env_count:
            normalized.extend({} for _ in range(env_count - len(normalized)))
        return normalized[:env_count]
    if isinstance(raw_info, dict):
        rows: List[Dict[str, Any]] = []
        for index in range(env_count):
            sliced: Dict[str, Any] = {}
            for key, value in raw_info.items():
                if not isinstance(key, str):
                    continue
                if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
                    try:
                        sliced[key] = _to_python_value(value[index])
                        continue
                    except Exception:
                        pass
                sliced[key] = _to_python_value(value)
            rows.append(sliced)
        return rows
    return [{} for _ in range(env_count)]


def _parse_step_output(step_output: Any, torch: Any, device: str) -> Tuple[Any, Any, Any, List[Dict[str, Any]]]:
    if not isinstance(step_output, tuple) or len(step_output) < 4:
        raise RuntimeError("Isaac step output shape is unsupported")
    observation = _extract_observation(step_output[0], torch, device)
    env_reward = _as_tensor(step_output[1], torch, device)
    done_tensor = _as_tensor(step_output[2], torch, device).bool()
    info_list = _normalize_info_list(step_output[3], int(env_reward.shape[0]))
    return observation, env_reward, done_tensor, info_list


class _ActionSpec:
    def __init__(self, kind: str, action_dim: int, low: Any = None, high: Any = None) -> None:
        self.kind = kind
        self.action_dim = action_dim
        self.low = low
        self.high = high


def _resolve_action_spec(environment: Any, torch: Any, device: str) -> _ActionSpec:
    action_space = getattr(environment, "action_space", None)
    discrete_size = getattr(action_space, "n", None)
    if isinstance(discrete_size, int) and discrete_size > 0:
        return _ActionSpec("discrete", discrete_size)

    shape = getattr(action_space, "shape", None)
    action_dim = 1
    if isinstance(shape, Sequence) and len(shape) > 0:
        action_dim = int(shape[-1])
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    low_tensor = _as_tensor(low, torch, device) if low is not None else None
    high_tensor = _as_tensor(high, torch, device) if high is not None else None
    return _ActionSpec("continuous", action_dim, low_tensor, high_tensor)


class _PolicyNetwork:
    def __init__(self, torch: Any, obs_dim: int, action_spec: _ActionSpec, device: str) -> None:
        self.torch = torch
        self.action_spec = action_spec
        hidden = 128
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        ).to(device)
        self.head = torch.nn.Linear(hidden, action_spec.action_dim).to(device)
        self.log_std = None
        if action_spec.kind != "discrete":
            self.log_std = torch.nn.Parameter(torch.zeros(action_spec.action_dim, device=device))

    def parameters(self) -> Any:
        if self.log_std is None:
            return list(self.model.parameters()) + list(self.head.parameters())
        return list(self.model.parameters()) + list(self.head.parameters()) + [self.log_std]

    def sample(self, observation: Any) -> Tuple[Any, Any]:
        features = self.model(observation)
        if self.action_spec.kind == "discrete":
            logits = self.head(features)
            distribution = self.torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action, log_prob

        mean = self.head(features)
        std = self.log_std.exp().expand_as(mean)
        distribution = self.torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        bounded = self.torch.tanh(raw_action)
        if self.action_spec.low is not None and self.action_spec.high is not None:
            low = self.action_spec.low
            high = self.action_spec.high
            bounded = low + ((bounded + 1.0) * 0.5 * (high - low))
        return bounded, log_prob

    def deterministic(self, observation: Any) -> Any:
        features = self.model(observation)
        if self.action_spec.kind == "discrete":
            return self.head(features).argmax(dim=-1)
        action = self.torch.tanh(self.head(features))
        if self.action_spec.low is not None and self.action_spec.high is not None:
            low = self.action_spec.low
            high = self.action_spec.high
            action = low + ((action + 1.0) * 0.5 * (high - low))
        return action


def _load_reward_callable(source_text: str, entrypoint_name: str) -> Any:
    namespace: Dict[str, Any] = {}
    exec(source_text, namespace, namespace)
    candidate = namespace.get(entrypoint_name)
    if candidate is None or not callable(candidate):
        raise RuntimeError("reward entrypoint %r was not defined as callable" % entrypoint_name)
    return candidate


def _build_reward_arguments(
    previous_observation: Any,
    next_observation: Any,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    action: Any,
    step_index: int,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "state": next_observation,
        "observation": next_observation,
        "next_observation": next_observation,
        "previous_observation": previous_observation,
        "env_reward": env_reward,
        "environment_reward": env_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action": action,
        "step_index": step_index,
        "info": info,
    }
    for key, value in info.items():
        if isinstance(key, str) and key.isidentifier() and key not in arguments:
            arguments[key] = value
    return arguments


def _select_reward_kwargs(parameters: Tuple[inspect.Parameter, ...], available: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    missing: List[str] = []
    accepts_var_keyword = False
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise RuntimeError("reward entrypoint uses unsupported positional-only parameters")
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        name = parameter.name
        if name in available:
            kwargs[name] = available[name]
            continue
        if parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "reward entrypoint requires unsupported parameters: %s"
            % ", ".join(repr(item) for item in missing)
        )
    if accepts_var_keyword:
        for name, value in available.items():
            kwargs.setdefault(name, value)
    return kwargs


def _score_batch(
    reward_callable: Any,
    previous_observation: Any,
    next_observation: Any,
    env_reward: Any,
    action: Any,
    done_tensor: Any,
    step_index: int,
    info_list: List[Dict[str, Any]],
    torch: Any,
) -> Any:
    parameters = tuple(inspect.signature(reward_callable).parameters.values())
    reward_values: List[float] = []
    env_count = int(next_observation.shape[0])
    for env_index in range(env_count):
        previous_item = _to_python_value(previous_observation[env_index])
        next_item = _to_python_value(next_observation[env_index])
        action_item = _to_python_value(action[env_index])
        env_reward_item = float(_to_python_value(env_reward[env_index]))
        done_item = bool(_to_python_value(done_tensor[env_index]))
        info_item = info_list[env_index] if env_index < len(info_list) else {}
        kwargs = _select_reward_kwargs(
            parameters=parameters,
            available=_build_reward_arguments(
                previous_observation=previous_item,
                next_observation=next_item,
                env_reward=env_reward_item,
                terminated=done_item,
                truncated=False,
                action=action_item,
                step_index=step_index,
                info=info_item,
            ),
        )
        value = reward_callable(**kwargs)
        if not isinstance(value, (int, float)):
            raise RuntimeError("reward entrypoint must return numeric values")
        reward_values.append(float(value))
    return torch.tensor(reward_values, dtype=torch.float32, device=next_observation.device)


def _policy_gradient_step(
    optimizer: Any,
    log_probs: List[Any],
    rewards: List[Any],
    dones: List[Any],
    gamma: float,
    torch: Any,
) -> None:
    if len(log_probs) == 0:
        return
    returns: List[Any] = []
    running = torch.zeros_like(rewards[-1])
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running = reward + gamma * running * (1.0 - done.float())
        returns.append(running)
    returns.reverse()
    stacked_returns = torch.stack(returns)
    normalized_returns = (stacked_returns - stacked_returns.mean()) / (
        stacked_returns.std(unbiased=False) + 1e-8
    )
    stacked_log_probs = torch.stack(log_probs)
    loss = -(stacked_log_probs * normalized_returns.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _first_numeric(info: Dict[str, Any], keys: Tuple[str, ...], default: float) -> float:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        python_value = _to_python_value(value)
        if isinstance(python_value, bool):
            return float(python_value)
        if isinstance(python_value, (int, float)):
            return float(python_value)
    return default


def _fitness_values_for_step(
    environment_id: str,
    env_reward: Any,
    info_list: List[Dict[str, Any]],
    done_tensor: Any,
    torch: Any,
) -> Any:
    task = environment_id.strip().lower()
    if task == "cartpole":
        return (~done_tensor.bool()).float()
    values: List[float] = []
    for index in range(int(env_reward.shape[0])):
        info = info_list[index] if index < len(info_list) else {}
        if task == "humanoid":
            values.append(
                _first_numeric(
                    info,
                    ("x_velocity", "forward_reward", "lin_vel_x", "progress"),
                    default=float(_to_python_value(env_reward[index])),
                )
            )
            continue
        if task == "allegrohand":
            values.append(
                _first_numeric(
                    info,
                    ("consecutive_successes", "successes", "success", "task_success"),
                    default=float(_to_python_value(env_reward[index])),
                )
            )
            continue
        values.append(float(_to_python_value(env_reward[index])))
    return torch.tensor(values, dtype=torch.float32, device=env_reward.device)


def _fitness_metric_name(environment_id: str) -> str:
    task = environment_id.strip().lower()
    if task == "cartpole":
        return "duration"
    if task == "humanoid":
        return "x_velocity"
    if task == "allegrohand":
        return "consecutive_successes"
    return "environment_reward"


def _evaluate_policy(
    environment: Any,
    environment_id: str,
    policy: _PolicyNetwork,
    evaluation_episodes: int,
    max_eval_steps: int,
    seed: Optional[int],
    checkpoint_index: int,
    torch: Any,
    device: str,
) -> float:
    episode_scores: List[float] = []
    for episode_index in range(max(evaluation_episodes, 1)):
        _ = _seed_for_index(
            base_seed=seed,
            offset=(checkpoint_index * 10000) + episode_index + 1,
        )
        observation = _extract_observation(environment.reset(), torch, device)
        cumulative = 0.0
        step_count = 0
        for _ in range(max_eval_steps):
            action = policy.deterministic(observation)
            step_output = environment.step(action)
            observation, env_reward, done_tensor, info_list = _parse_step_output(step_output, torch, device)
            fitness_values = _fitness_values_for_step(
                environment_id=environment_id,
                env_reward=env_reward,
                info_list=info_list,
                done_tensor=done_tensor,
                torch=torch,
            )
            cumulative += float(fitness_values.mean().item())
            step_count += 1
            if bool(done_tensor.all().item()):
                break
        episode_scores.append(0.0 if step_count == 0 else cumulative / step_count)
    return sum(episode_scores) / max(len(episode_scores), 1)


def _safe_close_environment(environment: Any) -> None:
    if hasattr(environment, "close"):
        environment.close()
        return
    gym_handle = getattr(environment, "gym", None)
    sim_handle = getattr(environment, "sim", None)
    if gym_handle is not None and sim_handle is not None and hasattr(gym_handle, "destroy_sim"):
        gym_handle.destroy_sim(sim_handle)


def _execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    execution_request = payload.get("execution_request", {})
    reward_program = payload.get("reward_program", {})
    policy_config = payload.get("policy_config", {})
    backend_config = payload.get("backend_config", {})

    environment_id = str(execution_request.get("environment_id", "")).strip()
    if not environment_id:
        raise RuntimeError("execution_request.environment_id must not be blank")

    cfg_dir_override = None
    raw_cfg = backend_config.get("cfg_dir_override")
    if isinstance(raw_cfg, str) and raw_cfg.strip():
        cfg_dir_override = raw_cfg

    torch, isaacgymenvs, cfg_dir, available_tasks = _load_isaac_runtime(cfg_dir_override=cfg_dir_override)
    if environment_id not in available_tasks:
        return {
            "status": "error",
            "error": "Task %r is not registered in isaacgym_task_map" % environment_id,
            "runtime_status": _runtime_status(
                ready=False,
                reason="Task %r is unavailable in this Isaac worker runtime" % environment_id,
                missing=["isaacgymenvs task registry"],
            ),
        }

    seed = execution_request.get("seed")
    n_envs = int(policy_config.get("n_envs", 8))
    configured_device = str(policy_config.get("device", "auto"))
    device = _resolve_device(configured_device, torch)
    environment = isaacgymenvs.make(
        seed=seed,
        task=environment_id,
        num_envs=max(n_envs, 1),
        sim_device=device,
        rl_device=device,
        headless=True,
        force_render=False,
        virtual_screen_capture=False,
        cfg_dir=cfg_dir,
    )

    try:
        reward_callable = _load_reward_callable(
            source_text=str(reward_program.get("source_text", "")),
            entrypoint_name=str(reward_program.get("entrypoint_name", "reward")),
        )
        total_timesteps = int(policy_config.get("total_timesteps", 50000))
        checkpoint_count = max(int(policy_config.get("checkpoint_count", 10)), 1)
        eval_runs = max(int(policy_config.get("evaluation_run_count", 5)), 1)
        eval_episodes_per_checkpoint = max(
            int(policy_config.get("evaluation_episodes_per_checkpoint", 1)),
            1,
        )
        gamma = float(policy_config.get("gamma", 0.99))
        learning_rate = float(policy_config.get("learning_rate", 3e-4))
        eval_steps_floor = max(int(policy_config.get("evaluation_max_steps_floor", 32)), 1)
        checkpoint_timesteps = max(total_timesteps // checkpoint_count, 1)

        initial_obs = _extract_observation(environment.reset(), torch, device)
        obs_dim = int(initial_obs.shape[-1])
        action_spec = _resolve_action_spec(environment, torch, device)
        policy = _PolicyNetwork(torch=torch, obs_dim=obs_dim, action_spec=action_spec, device=device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        global_step = 0
        checkpoint_scores: List[float] = []
        event_trace: List[Dict[str, Any]] = []
        observation = initial_obs
        previous_observation = initial_obs

        for checkpoint_index in range(checkpoint_count):
            log_probs: List[Any] = []
            rewards: List[Any] = []
            dones: List[Any] = []
            local_steps = 0
            while local_steps < checkpoint_timesteps:
                policy_observation = observation.detach().clone()
                action, log_prob = policy.sample(policy_observation)
                step_output = environment.step(action)
                next_observation, env_reward, done_tensor, info_list = _parse_step_output(step_output, torch, device)
                shaped_reward = _score_batch(
                    reward_callable=reward_callable,
                    previous_observation=previous_observation,
                    next_observation=next_observation,
                    env_reward=env_reward,
                    action=action,
                    done_tensor=done_tensor,
                    step_index=global_step + local_steps + 1,
                    info_list=info_list,
                    torch=torch,
                )
                log_probs.append(log_prob)
                rewards.append(shaped_reward)
                dones.append(done_tensor)
                previous_observation = observation
                observation = next_observation
                local_steps += int(action.shape[0])
            global_step += local_steps

            _policy_gradient_step(
                optimizer=optimizer,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                gamma=gamma,
                torch=torch,
            )
            checkpoint_score = _evaluate_policy(
                environment=environment,
                environment_id=environment_id,
                policy=policy,
                evaluation_episodes=eval_episodes_per_checkpoint,
                max_eval_steps=max(eval_steps_floor, checkpoint_timesteps // max(n_envs, 1)),
                seed=seed,
                checkpoint_index=checkpoint_index,
                torch=torch,
                device=device,
            )
            checkpoint_scores.append(checkpoint_score)
            event_trace.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_timesteps": checkpoint_timesteps * (checkpoint_index + 1),
                    "fitness_score": round(float(checkpoint_score), 6),
                }
            )

        aggregate_score = max(checkpoint_scores) if checkpoint_scores else 0.0
        metrics = {
            "episode_reward": round(float(aggregate_score), 6),
            "fitness_metric_name": _fitness_metric_name(environment_id),
            "fitness_metric_mean": round(float(aggregate_score), 6),
            "checkpoint_fitness": [round(float(value), 6) for value in checkpoint_scores],
            "train_timesteps": total_timesteps,
            "checkpoint_count": checkpoint_count,
            "evaluation_run_count": eval_runs,
            "evaluation_episodes_per_checkpoint": eval_episodes_per_checkpoint,
            "n_envs": n_envs,
            "device": device,
            "evaluation_protocol": "isaacgym_policy_gradient_max_checkpoint_fitness",
        }
        return {
            "status": "ok",
            "outcome": {
                "metrics": metrics,
                "event_trace": event_trace,
                "runtime_status": _runtime_status(
                    ready=True,
                    reason="Isaac Gym runtime is ready for task %r" % environment_id,
                    missing=[],
                ),
                "manifest_metadata": {
                    "entrypoint_name": str(reward_program.get("entrypoint_name", "reward")),
                    "reward_parameters": [],
                    "evaluation_protocol": "isaacgym_policy_gradient_max_checkpoint_fitness",
                    "checkpoint_count": checkpoint_count,
                },
            },
        }
    finally:
        _safe_close_environment(environment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Isaac Gym worker (py3.8)")
    parser.add_argument("--request", help="Path to JSON request payload")
    parser.add_argument("--response", help="Path to JSON response payload")
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run runtime health checks and emit JSON to stdout or --response.",
    )
    args = parser.parse_args()

    if args.healthcheck:
        payload = _healthcheck_payload()
        text = _safe_json_dump(payload)
        if args.response:
            Path(args.response).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0 if payload.get("status") == "ok" else 1

    if not args.request or not args.response:
        parser.error("--request and --response are required unless --healthcheck is used")

    response_path = Path(args.response)
    try:
        request_payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
        response_payload = _execute(request_payload)
        response_path.write_text(_safe_json_dump(response_payload), encoding="utf-8")
        return 0 if response_payload.get("status") == "ok" else 1
    except Exception as exc:  # pragma: no cover - defensive guard
        response_payload = {
            "status": "error",
            "error": "%s: %s" % (type(exc).__name__, exc),
            "runtime_status": _runtime_status(False, "Isaac worker execution failed", ["worker execution"]),
            "traceback": traceback.format_exc(),
        }
        response_path.write_text(_safe_json_dump(response_payload), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
