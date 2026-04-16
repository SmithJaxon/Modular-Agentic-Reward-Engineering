"""
Summary: Real Gymnasium experiment runner with rollout and Humanoid PPO evaluation modes.
Created: 2026-04-02
Last Updated: 2026-04-06
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol, cast

import gymnasium as gym

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.execution_service import (
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
)
from rewardlab.experiments.reward_program import RewardProgram
from rewardlab.schemas.runtime_status import BackendRuntimeStatus
from rewardlab.schemas.session_config import EnvironmentBackend


@dataclass(frozen=True)
class HumanoidPpoEvaluationConfig:
    """Configuration for Gymnasium Humanoid PPO evaluation."""

    enabled_environment_ids: frozenset[str] = field(
        default_factory=lambda: frozenset({"Humanoid-v4", "Humanoid-v5"})
    )
    total_timesteps: int = 50_000
    checkpoint_count: int = 10
    evaluation_run_count: int = 5
    evaluation_episodes_per_checkpoint: int = 1
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    @classmethod
    def from_environment(cls) -> HumanoidPpoEvaluationConfig:
        """Load PPO evaluation settings from environment variables."""

        return cls(
            total_timesteps=_int_from_env("REWARDLAB_PPO_TOTAL_TIMESTEPS", 50_000),
            checkpoint_count=_int_from_env("REWARDLAB_PPO_CHECKPOINT_COUNT", 10),
            evaluation_run_count=_int_from_env("REWARDLAB_PPO_EVAL_RUNS", 5),
            evaluation_episodes_per_checkpoint=_int_from_env(
                "REWARDLAB_PPO_EVAL_EPISODES",
                1,
            ),
            n_steps=_int_from_env("REWARDLAB_PPO_N_STEPS", 2048),
            batch_size=_int_from_env("REWARDLAB_PPO_BATCH_SIZE", 64),
            n_epochs=_int_from_env("REWARDLAB_PPO_N_EPOCHS", 10),
            learning_rate=_float_from_env("REWARDLAB_PPO_LEARNING_RATE", 3e-4),
            gamma=_float_from_env("REWARDLAB_PPO_GAMMA", 0.99),
            gae_lambda=_float_from_env("REWARDLAB_PPO_GAE_LAMBDA", 0.95),
            clip_range=_float_from_env("REWARDLAB_PPO_CLIP_RANGE", 0.2),
        )

    def supports_environment(self, environment_id: str) -> bool:
        """Return whether the PPO protocol applies to the requested environment."""

        return environment_id in self.enabled_environment_ids

    def checkpoint_timesteps(self) -> int:
        """Return the timesteps budget to train between metric checkpoints."""

        return max(self.total_timesteps // max(self.checkpoint_count, 1), 1)


class PredictiveTrainer(Protocol):
    """Minimal trainer surface required by the Humanoid PPO evaluator."""

    def learn(
        self,
        total_timesteps: int,
        *,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> Any:
        """Train the policy for the requested number of timesteps."""

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Return the action selected for the supplied observation."""


PpoTrainerFactory = Callable[[Any, int | None, HumanoidPpoEvaluationConfig], PredictiveTrainer]


class GymnasiumExperimentRunner:
    """Execute Gymnasium rollouts or paper-style Humanoid PPO evaluation."""

    def __init__(
        self,
        *,
        backend: GymnasiumBackend | None = None,
        default_max_episode_steps: int = 200,
        humanoid_ppo_config: HumanoidPpoEvaluationConfig | None = None,
        ppo_trainer_factory: PpoTrainerFactory | None = None,
    ) -> None:
        """Store the backend adapter plus optional PPO evaluation dependencies."""

        self.backend = backend or GymnasiumBackend()
        self.default_max_episode_steps = default_max_episode_steps
        self.humanoid_ppo_config = (
            humanoid_ppo_config or HumanoidPpoEvaluationConfig.from_environment()
        )
        self.ppo_trainer_factory = ppo_trainer_factory

    def __call__(
        self,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
    ) -> ExecutionOutcome:
        """Execute the configured Gymnasium evaluation path for the candidate."""

        runtime_status = self.backend.get_runtime_status(execution_request.environment_id)
        if not runtime_status.ready:
            raise ExecutionError(runtime_status.status_reason, runtime_status=runtime_status)

        if self.humanoid_ppo_config.supports_environment(execution_request.environment_id):
            return self._execute_humanoid_ppo(
                execution_request=execution_request,
                reward_program=reward_program,
                runtime_status=runtime_status,
            )
        return self._execute_rollout(
            execution_request=execution_request,
            reward_program=reward_program,
            runtime_status=runtime_status,
        )

    def _execute_rollout(
        self,
        *,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
        runtime_status: BackendRuntimeStatus,
    ) -> ExecutionOutcome:
        """Execute a single deterministic Gymnasium rollout."""

        environment = self.backend.create_environment(
            execution_request.environment_id,
            seed=execution_request.seed,
            render_mode=execution_request.render_mode,
        )
        step_limit = execution_request.max_episode_steps or self.default_max_episode_steps

        try:
            observation, _ = environment.reset(seed=execution_request.seed)
            episode_reward = 0.0
            environment_reward = 0.0
            terminated = False
            truncated = False
            event_trace: list[dict[str, Any]] = []

            for step_index in range(1, step_limit + 1):
                action = _select_default_action(environment=environment, observation=observation)
                next_observation, env_reward, terminated, truncated, info = environment.step(action)
                shaped_reward = _score_transition(
                    reward_program=reward_program,
                    previous_observation=observation,
                    next_observation=next_observation,
                    env_reward=env_reward,
                    terminated=terminated,
                    truncated=truncated,
                    action=action,
                    step_index=step_index,
                    info=info,
                )
                episode_reward += shaped_reward
                environment_reward += env_reward
                event_trace.append(
                    {
                        "step_index": step_index,
                        "action": action,
                        "environment_reward": env_reward,
                        "reward_program_output": shaped_reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break
        finally:
            self.backend.close_environment(environment)

        metrics = {
            "episode_reward": round(episode_reward, 6),
            "environment_reward": round(environment_reward, 6),
            "step_count": len(event_trace),
            "terminated": terminated,
            "truncated": truncated,
            "evaluation_protocol": "single_rollout",
        }
        return ExecutionOutcome(
            metrics=metrics,
            event_trace=event_trace,
            runtime_status=runtime_status,
            manifest_metadata={
                "entrypoint_name": reward_program.entrypoint_name,
                "reward_parameters": list(reward_program.parameter_names()),
                "evaluation_protocol": "single_rollout",
            },
        )

    def _execute_humanoid_ppo(
        self,
        *,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
        runtime_status: BackendRuntimeStatus,
    ) -> ExecutionOutcome:
        """Execute Humanoid PPO training and evaluate checkpoints by mean x velocity."""

        trainer_factory = self.ppo_trainer_factory or _default_ppo_trainer_factory()
        if trainer_factory is None:
            ppo_runtime_status = BackendRuntimeStatus(
                backend=EnvironmentBackend.GYMNASIUM,
                ready=False,
                status_reason=(
                    "stable_baselines3 is not installed in the active worktree-local .venv; "
                    "Gymnasium Humanoid PPO evaluation cannot run"
                ),
                missing_prerequisites=[
                    "install approved stable-baselines3 dependency inside .venv"
                ],
                detected_version=runtime_status.detected_version,
            )
            raise ExecutionError(
                ppo_runtime_status.status_reason,
                runtime_status=ppo_runtime_status,
            )

        checkpoint_timesteps = self.humanoid_ppo_config.checkpoint_timesteps()
        checkpoint_metrics: list[list[float]] = []
        event_trace: list[dict[str, Any]] = []
        per_run_best_metrics: list[float] = []

        for run_index in range(self.humanoid_ppo_config.evaluation_run_count):
            train_seed = _seed_for_run(execution_request.seed, run_index)
            raw_training_environment = self.backend.create_environment(
                execution_request.environment_id,
                seed=train_seed,
                render_mode=execution_request.render_mode,
            )
            evaluation_environment = self.backend.create_environment(
                execution_request.environment_id,
                seed=train_seed,
                render_mode=execution_request.render_mode,
            )
            training_environment = RewardFunctionEnvironment(
                environment=raw_training_environment,
                reward_program=reward_program,
            )
            try:
                trainer = trainer_factory(
                    training_environment,
                    train_seed,
                    self.humanoid_ppo_config,
                )
                run_checkpoint_metrics: list[float] = []
                for checkpoint_index in range(self.humanoid_ppo_config.checkpoint_count):
                    trainer.learn(
                        checkpoint_timesteps,
                        progress_bar=False,
                        reset_num_timesteps=(checkpoint_index == 0),
                    )
                    mean_x_velocity = _evaluate_policy_mean_x_velocity(
                        trainer=trainer,
                        environment=evaluation_environment,
                        episode_count=self.humanoid_ppo_config.evaluation_episodes_per_checkpoint,
                        seed=train_seed,
                    )
                    run_checkpoint_metrics.append(mean_x_velocity)
                    event_trace.append(
                        {
                            "run_index": run_index,
                            "checkpoint_index": checkpoint_index,
                            "checkpoint_timesteps": checkpoint_timesteps * (checkpoint_index + 1),
                            "mean_x_velocity": round(mean_x_velocity, 6),
                        }
                    )
                checkpoint_metrics.append(run_checkpoint_metrics)
                per_run_best_metrics.append(max(run_checkpoint_metrics))
            finally:
                self.backend.close_environment(evaluation_environment)
                self.backend.close_environment(training_environment)

        aggregate_score = sum(per_run_best_metrics) / max(len(per_run_best_metrics), 1)
        metrics = {
            "episode_reward": round(aggregate_score, 6),
            "fitness_metric_name": "mean_x_velocity",
            "fitness_metric_mean": round(aggregate_score, 6),
            "per_run_best_mean_x_velocity": [round(value, 6) for value in per_run_best_metrics],
            "checkpoint_mean_x_velocity": [
                [round(value, 6) for value in run_metrics]
                for run_metrics in checkpoint_metrics
            ],
            "train_timesteps": self.humanoid_ppo_config.total_timesteps,
            "checkpoint_count": self.humanoid_ppo_config.checkpoint_count,
            "evaluation_run_count": self.humanoid_ppo_config.evaluation_run_count,
            "evaluation_episodes_per_checkpoint": (
                self.humanoid_ppo_config.evaluation_episodes_per_checkpoint
            ),
            "evaluation_protocol": "humanoid_ppo_max_checkpoint_mean_x_velocity",
        }
        return ExecutionOutcome(
            metrics=metrics,
            event_trace=event_trace,
            runtime_status=runtime_status,
            manifest_metadata={
                "entrypoint_name": reward_program.entrypoint_name,
                "reward_parameters": list(reward_program.parameter_names()),
                "evaluation_protocol": "humanoid_ppo_max_checkpoint_mean_x_velocity",
                "checkpoint_count": self.humanoid_ppo_config.checkpoint_count,
                "evaluation_run_count": self.humanoid_ppo_config.evaluation_run_count,
            },
        )


class RewardFunctionEnvironment(gym.Env[Any, Any]):
    """Reward-shaping wrapper used for PPO training on Gymnasium environments."""

    def __init__(self, *, environment: Any, reward_program: RewardProgram) -> None:
        """Store the wrapped environment and reward program."""

        self.environment = environment
        self.reward_program = reward_program
        self.action_space = cast(Any, getattr(environment, "action_space", None))
        self.observation_space = cast(Any, getattr(environment, "observation_space", None))
        self.metadata = getattr(environment, "metadata", {})
        self._last_observation: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        """Reset the wrapped environment and preserve the current observation."""

        if options is None:
            observation, info = self.environment.reset(seed=seed)
        else:
            observation, info = self.environment.reset(seed=seed, options=options)
        self._last_observation = observation
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Replace the environment reward with the candidate reward program output."""

        next_observation, env_reward, terminated, truncated, info = self.environment.step(action)
        shaped_reward = _score_transition(
            reward_program=self.reward_program,
            previous_observation=self._last_observation,
            next_observation=next_observation,
            env_reward=env_reward,
            terminated=terminated,
            truncated=truncated,
            action=action,
            step_index=0,
            info=info,
        )
        self._last_observation = next_observation
        return next_observation, shaped_reward, terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment handle."""

        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    def render(self) -> Any:
        """Delegate rendering to the wrapped environment when available."""

        render = getattr(self.environment, "render", None)
        if callable(render):
            return render()
        return None

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""

        return getattr(self.environment, name)


def _select_default_action(*, environment: Any, observation: Any) -> Any:
    """Choose a deterministic default action for a Gymnasium environment."""

    action_space = getattr(environment, "action_space", None)
    discrete_size = getattr(action_space, "n", None)
    discrete_count = _coerce_int(discrete_size)
    if discrete_count == 2 and _is_numeric_sequence(
        observation,
        min_length=4,
    ):
        pole_angle = float(observation[2])
        pole_velocity = float(observation[3])
        return 1 if pole_angle + (pole_velocity * 0.1) > 0 else 0

    if discrete_count is not None and discrete_count > 0:
        return 0

    shape = getattr(action_space, "shape", None)
    if isinstance(shape, Sequence):
        return _zeros_from_shape(shape)
    return 0


def _score_transition(
    *,
    reward_program: RewardProgram,
    previous_observation: Any,
    next_observation: Any,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    action: Any,
    step_index: int,
    info: dict[str, Any],
) -> float:
    """Evaluate the reward program against a single environment transition."""

    reward_callable = reward_program.require_callable()
    parameters = inspect.signature(reward_callable).parameters
    available_arguments = _build_reward_arguments(
        previous_observation=previous_observation,
        next_observation=next_observation,
        env_reward=env_reward,
        terminated=terminated,
        truncated=truncated,
        action=action,
        step_index=step_index,
        info=info,
    )
    kwargs = _select_call_arguments(
        parameters=tuple(parameters.values()),
        available=available_arguments,
    )
    try:
        value = reward_callable(**kwargs)
    except TypeError as exc:
        raise ExecutionError(
            f"reward entrypoint {reward_program.entrypoint_name!r} could not be called: {exc}"
        ) from exc

    if not isinstance(value, (int, float)):
        raise ExecutionError(
            f"reward entrypoint {reward_program.entrypoint_name!r} must return a numeric value"
        )
    return float(value)


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
    """Build the named arguments available to a candidate reward function."""

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
    if _is_numeric_sequence(next_observation, min_length=4):
        arguments.update(
            {
                "cart_position": float(next_observation[0]),
                "cart_velocity": float(next_observation[1]),
                "pole_angle_radians": float(next_observation[2]),
                "angular_velocity": float(next_observation[3]),
            }
        )
    for key, value in info.items():
        if isinstance(key, str) and key.isidentifier() and key not in arguments:
            arguments[key] = value
    return arguments


def _select_call_arguments(
    *,
    parameters: tuple[inspect.Parameter, ...],
    available: dict[str, Any],
) -> dict[str, Any]:
    """Select the available keyword arguments needed by the reward callable."""

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    accepts_var_keyword = False
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise ExecutionError(
                "reward entrypoint uses positional-only parameters, which are not supported "
                "for Gymnasium rollout invocation"
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
        joined = ", ".join(repr(name) for name in missing)
        raise ExecutionError(
            f"reward entrypoint requires unsupported parameters for Gymnasium rollout: {joined}"
        )
    if accepts_var_keyword:
        for name, value in available.items():
            kwargs.setdefault(name, value)
    return kwargs


def _is_numeric_sequence(value: Any, *, min_length: int) -> bool:
    """Return whether the value looks like a finite-length numeric observation vector."""

    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence) and not _looks_indexable(value):
        return False
    try:
        if len(value) < min_length:
            return False
    except TypeError:
        return False
    try:
        return all(isinstance(value[index], Real) for index in range(min_length))
    except (IndexError, KeyError, TypeError):
        return False


def _looks_indexable(value: Any) -> bool:
    """Return whether a value exposes basic positional indexing operations."""

    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _zeros_from_shape(shape: Sequence[Any]) -> Any:
    """Construct a nested zero structure matching a simple shape tuple."""

    normalized = [int(item) for item in shape]
    if not normalized:
        return 0.0
    if len(normalized) == 1:
        return [0.0] * max(normalized[0], 1)
    return [_zeros_from_shape(normalized[1:]) for _ in range(max(normalized[0], 1))]


def _evaluate_policy_mean_x_velocity(
    *,
    trainer: PredictiveTrainer,
    environment: Any,
    episode_count: int,
    seed: int | None,
) -> float:
    """Evaluate a policy on raw Gymnasium Humanoid and return mean x velocity."""

    episode_metrics: list[float] = []
    for episode_index in range(episode_count):
        episode_seed = _seed_for_run(seed, episode_index)
        observation, _ = environment.reset(seed=episode_seed)
        terminated = False
        truncated = False
        metric_total = 0.0
        step_count = 0
        while not terminated and not truncated:
            action, _ = trainer.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            metric_total += _extract_humanoid_fitness_metric(info)
            step_count += 1
        if step_count == 0:
            episode_metrics.append(0.0)
        else:
            episode_metrics.append(metric_total / step_count)
    return sum(episode_metrics) / max(len(episode_metrics), 1)


def _extract_humanoid_fitness_metric(info: dict[str, Any]) -> float:
    """Extract the Humanoid task fitness metric from Gymnasium step info."""

    x_velocity = info.get("x_velocity")
    if isinstance(x_velocity, (int, float)):
        return float(x_velocity)
    forward_reward = info.get("forward_reward")
    if isinstance(forward_reward, (int, float)):
        return float(forward_reward)
    return 0.0


def _default_ppo_trainer_factory() -> PpoTrainerFactory | None:
    """Return the default SB3 PPO trainer factory when the dependency is present."""

    try:
        from stable_baselines3 import PPO  # type: ignore[import-not-found]
    except Exception:
        return None

    def build_trainer(
        environment: Any,
        seed: int | None,
        config: HumanoidPpoEvaluationConfig,
    ) -> PredictiveTrainer:
        """Construct a stable-baselines3 PPO trainer for Humanoid experiments."""

        return cast(
            PredictiveTrainer,
            PPO(
                "MlpPolicy",
                environment,
                seed=seed,
                verbose=0,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                device="auto",
            ),
        )

    return build_trainer


def _seed_for_run(base_seed: int | None, index: int) -> int | None:
    """Derive a stable per-run seed from an optional base seed."""

    if base_seed is None:
        return None
    return base_seed + index


def _int_from_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback value."""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return max(parsed, 1)


def _float_from_env(name: str, default: float) -> float:
    """Read a float environment variable with a fallback value."""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _coerce_int(value: object) -> int | None:
    """Return an integer when the provided object can be safely coerced."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except ValueError:
            return None

    converter = getattr(value, "__int__", None)
    if not callable(converter):
        return None
    converted = converter()
    if isinstance(converted, int):
        return converted
    return None
