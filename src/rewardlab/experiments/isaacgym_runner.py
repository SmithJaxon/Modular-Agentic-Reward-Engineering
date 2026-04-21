"""
Summary: Isaac Gym experiment runner with stochastic policy optimization.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.experiments.execution_service import (
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
)
from rewardlab.experiments.reward_program import RewardProgram


@dataclass(frozen=True)
class IsaacGymPolicyConfig:
    """Policy-search configuration for Isaac Gym experiments."""

    total_timesteps: int = 50_000
    checkpoint_count: int = 10
    evaluation_run_count: int = 5
    evaluation_episodes_per_checkpoint: int = 1
    n_envs: int = 8
    device: str = "auto"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    evaluation_max_steps_floor: int = 32

    def checkpoint_timesteps(self) -> int:
        """Return training timesteps consumed between checkpoints."""

        return max(self.total_timesteps // max(self.checkpoint_count, 1), 1)


@dataclass(frozen=True)
class IsaacGymSubprocessConfig:
    """Configuration for per-run Isaac process isolation."""

    enabled: bool = True
    timeout_seconds: int = 300
    stderr_tail_lines: int = 80
    worker_command: str | None = None


class IsaacGymExperimentRunner:
    """Execute Isaac Gym candidate runs with stochastic policy optimization."""

    def __init__(
        self,
        *,
        backend: IsaacGymBackend | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        policy_config: IsaacGymPolicyConfig | None = None,
        subprocess_config: IsaacGymSubprocessConfig | None = None,
    ) -> None:
        """Store backend/runtime configuration used for candidate execution."""

        resolved_backend_kwargs = backend_kwargs or {}
        self.backend = backend or IsaacGymBackend(**resolved_backend_kwargs)
        self.policy_config = policy_config or IsaacGymPolicyConfig()
        self.subprocess_config = subprocess_config or _subprocess_config_from_environment()

    def __call__(
        self,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
    ) -> ExecutionOutcome:
        """Execute one Isaac Gym run and return normalized metrics plus event trace.

        Isaac Gym Preview 4 mutates global module/runtime state and is not reliably re-entrant
        across many runs in one interpreter. Run each execution in an isolated subprocess by
        default so candidate, comparison, and probe runs do not contaminate each other.
        """

        if self.subprocess_config.enabled:
            return _run_isolated_subprocess(
                execution_request=execution_request,
                reward_program=reward_program,
                policy_config=self.policy_config,
                subprocess_config=self.subprocess_config,
                backend=self.backend,
            )
        return _execute_in_process(
            execution_request=execution_request,
            reward_program=reward_program,
            backend=self.backend,
            policy_config=self.policy_config,
        )


def _execute_in_process(
    *,
    execution_request: ExecutionRequest,
    reward_program: RewardProgram,
    backend: IsaacGymBackend,
    policy_config: IsaacGymPolicyConfig,
    close_environment: bool = True,
) -> ExecutionOutcome:
    """Execute one Isaac Gym run inside the current Python process."""

    runtime_status = backend.get_runtime_status(execution_request.environment_id)
    if not runtime_status.ready:
        raise ExecutionError(runtime_status.status_reason, runtime_status=runtime_status)

    torch = _require_torch()
    device = _resolve_device(policy_config.device, torch)
    environment = backend.create_task_environment(
        environment_id=execution_request.environment_id,
        seed=execution_request.seed,
        num_envs=policy_config.n_envs,
        sim_device=device,
        rl_device=device,
        headless=True,
    )

    try:
        initial_obs = _extract_observation_tensor(environment.reset(), torch, device)
        obs_dim = int(initial_obs.shape[-1])
        action_spec = _resolve_action_spec(environment, torch, device)
        policy = _PolicyNetwork(
            torch=torch,
            obs_dim=obs_dim,
            action_spec=action_spec,
            device=device,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=policy_config.learning_rate)
        checkpoint_timesteps = policy_config.checkpoint_timesteps()
        global_step = 0
        checkpoint_scores: list[float] = []
        event_trace: list[dict[str, Any]] = []

        observation = initial_obs
        previous_observation = initial_obs
        for checkpoint_index in range(policy_config.checkpoint_count):
            log_probs: list[Any] = []
            rewards: list[Any] = []
            dones: list[Any] = []
            local_steps = 0
            while local_steps < checkpoint_timesteps:
                policy_observation = observation.detach().clone()
                action, log_prob = policy.sample(policy_observation)
                next_raw = environment.step(action)
                (
                    next_observation,
                    env_reward,
                    done_tensor,
                    info_list,
                ) = _parse_step_output(next_raw, torch, device)
                shaped_reward = _score_batch(
                    reward_program=reward_program,
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
                gamma=policy_config.gamma,
                torch=torch,
            )
            checkpoint_score = _evaluate_policy_fitness(
                environment=environment,
                environment_id=execution_request.environment_id,
                policy=policy,
                evaluation_episodes=policy_config.evaluation_episodes_per_checkpoint,
                max_eval_steps=max(
                    max(policy_config.evaluation_max_steps_floor, 1),
                    checkpoint_timesteps // max(policy_config.n_envs, 1),
                ),
                seed=execution_request.seed,
                checkpoint_index=checkpoint_index,
                torch=torch,
                device=device,
            )
            checkpoint_scores.append(checkpoint_score)
            event_trace.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_timesteps": checkpoint_timesteps * (checkpoint_index + 1),
                    "fitness_score": round(checkpoint_score, 6),
                }
            )

        aggregate_score = max(checkpoint_scores) if checkpoint_scores else 0.0
        metrics = {
            "episode_reward": round(float(aggregate_score), 6),
            "fitness_metric_name": _fitness_metric_name(execution_request.environment_id),
            "fitness_metric_mean": round(float(aggregate_score), 6),
            "checkpoint_fitness": [round(float(value), 6) for value in checkpoint_scores],
            "train_timesteps": policy_config.total_timesteps,
            "checkpoint_count": policy_config.checkpoint_count,
            "evaluation_run_count": policy_config.evaluation_run_count,
            "evaluation_episodes_per_checkpoint": (
                policy_config.evaluation_episodes_per_checkpoint
            ),
            "n_envs": policy_config.n_envs,
            "device": device,
            "evaluation_protocol": "isaacgym_policy_gradient_max_checkpoint_fitness",
        }
        return ExecutionOutcome(
            metrics=metrics,
            event_trace=event_trace,
            runtime_status=runtime_status,
            manifest_metadata={
                "entrypoint_name": reward_program.entrypoint_name,
                "reward_parameters": list(reward_program.parameter_names()),
                "evaluation_protocol": "isaacgym_policy_gradient_max_checkpoint_fitness",
                "checkpoint_count": policy_config.checkpoint_count,
            },
        )
    finally:
        if close_environment:
            _safe_close_environment(backend, environment)


def _run_isolated_subprocess(
    *,
    execution_request: ExecutionRequest,
    reward_program: RewardProgram,
    policy_config: IsaacGymPolicyConfig,
    subprocess_config: IsaacGymSubprocessConfig,
    backend: IsaacGymBackend,
) -> ExecutionOutcome:
    """Execute one Isaac Gym run in a fresh Python interpreter."""

    payload = {
        "execution_request": {
            "run_id": execution_request.run_id,
            "backend": execution_request.backend.value,
            "environment_id": execution_request.environment_id,
            "run_type": execution_request.run_type.value,
            "execution_mode": execution_request.execution_mode.value,
            "variant_label": execution_request.variant_label,
            "seed": execution_request.seed,
            "entrypoint_name": execution_request.entrypoint_name,
            "render_mode": execution_request.render_mode,
            "max_episode_steps": execution_request.max_episode_steps,
        },
        "reward_program": {
            "candidate_id": reward_program.candidate_id,
            "source_text": reward_program.source_text,
            "entrypoint_name": reward_program.entrypoint_name,
        },
        "policy_config": asdict(policy_config),
        "backend_config": {
            "cfg_dir_override": getattr(backend, "_cfg_dir_override", None),
        },
    }
    with tempfile.TemporaryDirectory(prefix="rewardlab_isaac_run_") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        response_path = Path(temp_dir) / "response.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")

        try:
            worker_cmd = resolve_worker_command(subprocess_config)
            completed = subprocess.run(  # noqa: S603 - worker command is an explicit config/env path
                worker_cmd
                + [
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                capture_output=True,
                text=True,
                timeout=max(subprocess_config.timeout_seconds, 30),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_tail = _tail_lines(exc.stdout or "", subprocess_config.stderr_tail_lines)
            stderr_tail = _tail_lines(exc.stderr or "", subprocess_config.stderr_tail_lines)
            raise ExecutionError(
                "Isaac Gym isolated run timed out "
                f"(timeout_seconds={subprocess_config.timeout_seconds}). "
                f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}"
            ) from exc
        if not response_path.exists():
            stdout_tail = _tail_lines(completed.stdout, subprocess_config.stderr_tail_lines)
            stderr_tail = _tail_lines(completed.stderr, subprocess_config.stderr_tail_lines)
            raise ExecutionError(
                "Isaac Gym isolated run process completed without response payload. "
                f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}"
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if completed.returncode != 0:
            stdout_tail = _tail_lines(completed.stdout, subprocess_config.stderr_tail_lines)
            stderr_tail = _tail_lines(completed.stderr, subprocess_config.stderr_tail_lines)
            worker_traceback = response.get("traceback")
            message = (
                "Isaac Gym isolated run process exited non-zero "
                f"(exit={completed.returncode}). stdout tail:\n{stdout_tail}\n\n"
                f"stderr tail:\n{stderr_tail}"
            )
            if worker_traceback:
                message = f"{message}\n\nworker traceback:\n{worker_traceback}"
            raise ExecutionError(message)

    if response.get("status") != "ok":
        runtime_status = response.get("runtime_status")
        raise ExecutionError(
            response.get("error", "Isaac Gym isolated run returned an unknown error"),
            runtime_status=(
                None
                if runtime_status is None
                else _runtime_status_from_payload(runtime_status)
            ),
        )
    outcome_payload = response.get("outcome", {})
    runtime_status_payload = outcome_payload.get("runtime_status")
    return ExecutionOutcome(
        metrics=outcome_payload.get("metrics", {}),
        event_trace=outcome_payload.get("event_trace"),
        runtime_status=(
            None
            if runtime_status_payload is None
            else _runtime_status_from_payload(runtime_status_payload)
        ),
        manifest_metadata=outcome_payload.get("manifest_metadata", {}),
    )


def _subprocess_config_from_environment() -> IsaacGymSubprocessConfig:
    """Resolve subprocess isolation settings from environment with safe defaults."""

    enabled_raw = os.getenv("REWARDLAB_ISAAC_ISOLATION_ENABLED", "1").strip().lower()
    enabled = enabled_raw not in {"0", "false", "no", "off"}
    timeout_raw = os.getenv("REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS", "").strip()
    timeout = 300
    if timeout_raw:
        try:
            timeout = max(int(timeout_raw), 30)
        except ValueError:
            timeout = 300
    worker_command = os.getenv("REWARDLAB_ISAAC_WORKER_COMMAND", "").strip() or None
    return IsaacGymSubprocessConfig(
        enabled=enabled,
        timeout_seconds=timeout,
        worker_command=worker_command,
    )


def resolve_worker_command(config: IsaacGymSubprocessConfig) -> list[str]:
    """Resolve subprocess command used to launch the isolated Isaac worker."""

    if config.worker_command is not None and config.worker_command.strip():
        return shlex.split(config.worker_command)
    explicit_python = os.getenv("REWARDLAB_ISAAC_WORKER_PYTHON", "").strip()
    python_bin = explicit_python or sys.executable
    return [python_bin, "-m", "rewardlab.experiments.isaacgym_worker"]


def _runtime_status_from_payload(payload: dict[str, Any]) -> Any:
    """Convert JSON payload back into BackendRuntimeStatus."""

    from rewardlab.schemas.runtime_status import BackendRuntimeStatus

    return BackendRuntimeStatus.model_validate(payload)


def _tail_lines(text: str | bytes | bytearray | None, line_count: int) -> str:
    """Return the last N lines from subprocess output for compact error surfaces."""

    if text is None or text == "":
        return "<empty>"
    if isinstance(text, (bytes, bytearray)):
        normalized = text.decode("utf-8", errors="replace")
    else:
        normalized = str(text)
    lines = normalized.splitlines()
    if len(lines) <= line_count:
        return "\n".join(lines)
    return "\n".join(lines[-line_count:])


class _ActionSpec:
    """Resolved action-space metadata for policy parameterization."""

    def __init__(
        self,
        *,
        kind: str,
        action_dim: int,
        low: Any | None = None,
        high: Any | None = None,
    ) -> None:
        """Store normalized action-space details."""

        self.kind = kind
        self.action_dim = action_dim
        self.low = low
        self.high = high


class _PolicyNetwork:
    """Simple two-layer policy used for stochastic policy-gradient updates."""

    def __init__(
        self,
        *,
        torch: Any,
        obs_dim: int,
        action_spec: _ActionSpec,
        device: str,
    ) -> None:
        """Construct the policy module and distribution parameters."""

        self.torch = torch
        self.action_spec = action_spec
        self.device = device
        hidden = 128
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        ).to(device)
        if action_spec.kind == "discrete":
            self.head = torch.nn.Linear(hidden, action_spec.action_dim).to(device)
            self.log_std = None
        else:
            self.head = torch.nn.Linear(hidden, action_spec.action_dim).to(device)
            self.log_std = torch.nn.Parameter(torch.zeros(action_spec.action_dim, device=device))

    def parameters(self) -> Any:
        """Return all trainable parameters for optimizer wiring."""

        if self.log_std is None:
            return list(self.model.parameters()) + list(self.head.parameters())
        return list(self.model.parameters()) + list(self.head.parameters()) + [self.log_std]

    def sample(self, observation: Any) -> tuple[Any, Any]:
        """Sample one stochastic action batch and return action + log probabilities."""

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
        """Return deterministic actions for evaluation."""

        features = self.model(observation)
        if self.action_spec.kind == "discrete":
            logits = self.head(features)
            return logits.argmax(dim=-1)
        action = self.torch.tanh(self.head(features))
        if self.action_spec.low is not None and self.action_spec.high is not None:
            low = self.action_spec.low
            high = self.action_spec.high
            action = low + ((action + 1.0) * 0.5 * (high - low))
        return action


def _score_batch(
    *,
    reward_program: RewardProgram,
    previous_observation: Any,
    next_observation: Any,
    env_reward: Any,
    action: Any,
    done_tensor: Any,
    step_index: int,
    info_list: list[dict[str, Any]],
    torch: Any,
) -> Any:
    """Evaluate reward-program outputs for one vectorized step."""

    if not hasattr(torch, "compile"):
        torch.compile = lambda fn=None, *args, **kwargs: fn  # type: ignore[attr-defined]

    reward_callable = reward_program.require_callable()
    parameters = tuple(inspect.signature(reward_callable).parameters.values())
    reward_values: list[float] = []
    env_count = int(next_observation.shape[0])
    for env_index in range(env_count):
        previous_item = _to_python_value(previous_observation[env_index], torch)
        next_item = _to_python_value(next_observation[env_index], torch)
        action_item = _to_python_value(action[env_index], torch)
        env_reward_item = float(_to_python_value(env_reward[env_index], torch))
        done_item = bool(_to_python_value(done_tensor[env_index], torch))
        info_item = info_list[env_index] if env_index < len(info_list) else {}
        kwargs = _select_call_arguments(
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
        try:
            value = reward_callable(**kwargs)
        except TypeError as exc:  # pragma: no cover - guarded by caller tests
            raise ExecutionError(
                f"reward entrypoint {reward_program.entrypoint_name!r} could not be called: {exc}"
            ) from exc
        if not isinstance(value, (int, float)):
            raise ExecutionError(
                f"reward entrypoint {reward_program.entrypoint_name!r} must return numeric values"
            )
        reward_values.append(float(value))
    return torch.tensor(reward_values, dtype=torch.float32, device=next_observation.device)


def _policy_gradient_step(
    *,
    optimizer: Any,
    log_probs: list[Any],
    rewards: list[Any],
    dones: list[Any],
    gamma: float,
    torch: Any,
) -> None:
    """Run one REINFORCE update from collected vectorized rollout segments."""

    if len(log_probs) == 0:
        return
    returns: list[Any] = []
    running = torch.zeros_like(rewards[-1])
    for reward, done in zip(reversed(rewards), reversed(dones), strict=False):
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


def _evaluate_policy_fitness(
    *,
    environment: Any,
    environment_id: str,
    policy: _PolicyNetwork,
    evaluation_episodes: int,
    max_eval_steps: int,
    seed: int | None,
    checkpoint_index: int,
    torch: Any,
    device: str,
) -> float:
    """Evaluate deterministic policy fitness over fixed-horizon vectorized rollouts."""

    episode_scores: list[float] = []
    for episode_index in range(max(evaluation_episodes, 1)):
        episode_seed = _seed_for_index(
            base_seed=seed,
            offset=(checkpoint_index * 10_000) + episode_index + 1,
        )
        observation = _extract_observation_tensor(environment.reset(), torch, device)
        cumulative = 0.0
        step_count = 0
        for _ in range(max_eval_steps):
            action = policy.deterministic(observation)
            step_output = environment.step(action)
            observation, env_reward, done_tensor, info_list = _parse_step_output(
                step_output,
                torch,
                device,
            )
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
        if step_count == 0:
            episode_scores.append(0.0)
        else:
            episode_scores.append(cumulative / step_count)
        del episode_seed
    return sum(episode_scores) / max(len(episode_scores), 1)


def _fitness_values_for_step(
    *,
    environment_id: str,
    env_reward: Any,
    info_list: list[dict[str, Any]],
    done_tensor: Any,
    torch: Any,
) -> Any:
    """Return per-environment task-fitness values for one step."""

    task = environment_id.strip().lower()
    if task == "cartpole":
        return (~done_tensor.bool()).float()

    values: list[float] = []
    for index in range(int(env_reward.shape[0])):
        info = info_list[index] if index < len(info_list) else {}
        if task == "humanoid":
            values.append(
                _first_numeric(
                    info,
                    ("x_velocity", "forward_reward", "lin_vel_x", "progress"),
                    default=float(_to_python_value(env_reward[index], torch)),
                    torch=torch,
                )
            )
            continue
        if task == "allegrohand":
            values.append(
                _first_numeric(
                    info,
                    ("consecutive_successes", "successes", "success", "task_success"),
                    default=float(_to_python_value(env_reward[index], torch)),
                    torch=torch,
                )
            )
            continue
        values.append(float(_to_python_value(env_reward[index], torch)))
    return torch.tensor(values, dtype=torch.float32, device=env_reward.device)


def _fitness_metric_name(environment_id: str) -> str:
    """Return reported fitness-metric name based on task id."""

    task = environment_id.strip().lower()
    if task == "cartpole":
        return "duration"
    if task == "humanoid":
        return "x_velocity"
    if task == "allegrohand":
        return "consecutive_successes"
    return "environment_reward"


def _build_reward_arguments(
    *,
    previous_observation: Any,
    next_observation: Any,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    action: Any,
    step_index: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Build available named arguments for reward-call invocation."""

    arguments: dict[str, Any] = {
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


def _select_call_arguments(
    *,
    parameters: tuple[inspect.Parameter, ...],
    available: dict[str, Any],
) -> dict[str, Any]:
    """Select keyword arguments needed by the reward callable signature."""

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    accepts_var_keyword = False
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise ExecutionError(
                "reward entrypoint uses positional-only parameters, which are not supported "
                "for Isaac Gym invocation"
            )
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
        joined = ", ".join(repr(item) for item in missing)
        raise ExecutionError(
            f"reward entrypoint requires unsupported parameters for Isaac Gym: {joined}"
        )
    if accepts_var_keyword:
        for name, value in available.items():
            kwargs.setdefault(name, value)
    return kwargs


def _parse_step_output(
    raw_step_output: Any,
    torch: Any,
    device: str,
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    """Normalize one IsaacGymEnvs step output payload."""

    if not isinstance(raw_step_output, tuple) or len(raw_step_output) < 4:
        raise ExecutionError("Isaac Gym step output shape is unsupported")
    observation = _extract_observation_tensor(raw_step_output[0], torch, device)
    env_reward = _as_tensor(raw_step_output[1], torch, device)
    done_tensor = _as_tensor(raw_step_output[2], torch, device).bool()
    info_list = _normalize_info_list(raw_step_output[3], int(env_reward.shape[0]), torch)
    return observation, env_reward, done_tensor, info_list


def _extract_observation_tensor(raw_observation: Any, torch: Any, device: str) -> Any:
    """Extract the primary observation tensor from IsaacGymEnvs outputs."""

    if isinstance(raw_observation, dict):
        candidate = raw_observation.get("obs")
        if candidate is None:
            raise ExecutionError("Isaac Gym observation dictionary is missing 'obs'")
        tensor = _as_tensor(candidate, torch, device)
    else:
        tensor = _as_tensor(raw_observation, torch, device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _as_tensor(value: Any, torch: Any, device: str) -> Any:
    """Convert arbitrary numeric payloads to floating-point torch tensors."""

    if hasattr(value, "to") and hasattr(value, "dtype"):
        return value.to(device).float()
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _normalize_info_list(raw_info: Any, env_count: int, torch: Any) -> list[dict[str, Any]]:
    """Return one per-environment info dictionary from backend step output."""

    if isinstance(raw_info, list):
        normalized = [item if isinstance(item, dict) else {} for item in raw_info]
        if len(normalized) < env_count:
            normalized.extend({} for _ in range(env_count - len(normalized)))
        return normalized[:env_count]
    if isinstance(raw_info, dict):
        return [_slice_info_entry(raw_info, index, torch) for index in range(env_count)]
    return [{} for _ in range(env_count)]


def _slice_info_entry(info_dict: dict[str, Any], index: int, torch: Any) -> dict[str, Any]:
    """Extract one environment-specific dictionary from vectorized info payloads."""

    sliced: dict[str, Any] = {}
    for key, value in info_dict.items():
        if not isinstance(key, str):
            continue
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
            try:
                sliced[key] = _to_python_value(value[index], torch)
                continue
            except (IndexError, KeyError, TypeError):
                ...
        sliced[key] = _to_python_value(value, torch)
    return sliced


def _resolve_action_spec(environment: Any, torch: Any, device: str) -> _ActionSpec:
    """Resolve action-space type and dimensionality from environment metadata."""

    action_space = getattr(environment, "action_space", None)
    discrete_size = getattr(action_space, "n", None)
    if isinstance(discrete_size, int) and discrete_size > 0:
        return _ActionSpec(kind="discrete", action_dim=discrete_size)

    shape = getattr(action_space, "shape", None)
    action_dim = 1
    if isinstance(shape, Sequence) and len(shape) > 0:
        action_dim = int(shape[-1])
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    low_tensor = _as_tensor(low, torch, device) if low is not None else None
    high_tensor = _as_tensor(high, torch, device) if high is not None else None
    return _ActionSpec(kind="continuous", action_dim=action_dim, low=low_tensor, high=high_tensor)


def _first_numeric(
    info: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: float,
    torch: Any,
) -> float:
    """Return the first available numeric info value across candidate keys."""

    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        python_value = _to_python_value(value, torch)
        if isinstance(python_value, bool):
            return float(python_value)
        if isinstance(python_value, (int, float)):
            return float(python_value)
    return default


def _to_python_value(value: Any, torch: Any) -> Any:
    """Convert torch-like values to Python scalars/lists recursively."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    if isinstance(value, dict):
        return {key: _to_python_value(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python_value(item, torch) for item in value]
    return value


def _resolve_device(configured: str, torch: Any) -> str:
    """Resolve an executable torch device string from configured value."""

    normalized = configured.strip().lower()
    if normalized == "auto":
        return "cuda:0" if _cuda_available(torch) else "cpu"
    if normalized.startswith("cuda") and not _cuda_available(torch):
        raise ExecutionError("execution.ppo.device requests CUDA but no CUDA runtime is available")
    return configured


def _cuda_available(torch: Any) -> bool:
    """Return whether the provided torch-like module exposes a CUDA runtime."""

    cuda_module = getattr(torch, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return False


def _require_torch() -> Any:
    """Import torch or raise a runtime-ready execution error."""

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import guard
        raise ExecutionError(
            "torch is required for Isaac Gym policy optimization but is not installed"
        ) from exc
    return torch


def _seed_for_index(*, base_seed: int | None, offset: int) -> int | None:
    """Return a stable seed derived from base seed and rollout offset."""

    if base_seed is None:
        return None
    return base_seed + offset


def _safe_close_environment(backend: IsaacGymBackend, environment: Any) -> None:
    """Close Isaac Gym environments when the runtime exposes a close method."""

    if hasattr(environment, "close"):
        backend.close_environment(environment)
        return
    gym_handle = getattr(environment, "gym", None)
    sim_handle = getattr(environment, "sim", None)
    if gym_handle is not None and sim_handle is not None and hasattr(gym_handle, "destroy_sim"):
        gym_handle.destroy_sim(sim_handle)
