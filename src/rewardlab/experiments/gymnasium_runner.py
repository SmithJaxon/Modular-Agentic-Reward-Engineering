"""
Summary: Minimal real Gymnasium experiment runner for RewardLab candidate evaluation.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.execution_service import (
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
)
from rewardlab.experiments.reward_program import RewardProgram


class GymnasiumExperimentRunner:
    """Execute one real Gymnasium rollout and score it with a reward program."""

    def __init__(
        self,
        *,
        backend: GymnasiumBackend | None = None,
        default_max_episode_steps: int = 200,
    ) -> None:
        """Store the backend adapter and default rollout budget."""

        self.backend = backend or GymnasiumBackend()
        self.default_max_episode_steps = default_max_episode_steps

    def __call__(
        self,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
    ) -> ExecutionOutcome:
        """Execute a single Gymnasium rollout for the supplied reward program."""

        runtime_status = self.backend.get_runtime_status(execution_request.environment_id)
        if not runtime_status.ready:
            raise ExecutionError(runtime_status.status_reason, runtime_status=runtime_status)

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
                action = _select_cartpole_action(observation)
                next_observation, env_reward, terminated, truncated, info = environment.step(action)
                shaped_reward = _score_transition(
                    reward_program=reward_program,
                    observation=next_observation,
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
        }
        return ExecutionOutcome(
            metrics=metrics,
            event_trace=event_trace,
            runtime_status=runtime_status,
            manifest_metadata={
                "entrypoint_name": reward_program.entrypoint_name,
                "reward_parameters": list(reward_program.parameter_names()),
            },
        )


def _select_cartpole_action(observation: Any) -> int:
    """Choose a simple deterministic action from a CartPole-style observation."""

    if _is_numeric_sequence(observation, min_length=4):
        pole_angle = float(observation[2])
        pole_velocity = float(observation[3])
        return 1 if pole_angle + (pole_velocity * 0.1) > 0 else 0
    return 0


def _score_transition(
    *,
    reward_program: RewardProgram,
    observation: Any,
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
        observation=observation,
        env_reward=env_reward,
        terminated=terminated,
        truncated=truncated,
        action=action,
        step_index=step_index,
        info=info,
    )
    kwargs = _select_call_arguments(parameters=tuple(parameters), available=available_arguments)
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
    observation: Any,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    action: Any,
    step_index: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Build the named arguments available to a candidate reward function."""

    arguments: dict[str, Any] = {
        "state": observation,
        "observation": observation,
        "env_reward": env_reward,
        "environment_reward": env_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action": action,
        "step_index": step_index,
        "info": info,
    }
    if _is_numeric_sequence(observation, min_length=4):
        arguments.update(
            {
                "cart_position": float(observation[0]),
                "cart_velocity": float(observation[1]),
                "pole_angle_radians": float(observation[2]),
                "angular_velocity": float(observation[3]),
            }
        )
    return arguments


def _select_call_arguments(
    *,
    parameters: tuple[str, ...],
    available: dict[str, Any],
) -> dict[str, Any]:
    """Select the available keyword arguments needed by the reward callable."""

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name in parameters:
        if name in available:
            kwargs[name] = available[name]
        else:
            missing.append(name)
    if missing:
        joined = ", ".join(repr(name) for name in missing)
        raise ExecutionError(
            f"reward entrypoint requires unsupported parameters for Gymnasium rollout: {joined}"
        )
    return kwargs


def _is_numeric_sequence(value: Any, *, min_length: int) -> bool:
    """Return whether the value looks like a finite-length numeric observation vector."""

    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    if len(value) < min_length:
        return False
    return all(isinstance(item, (int, float)) for item in value[:min_length])
