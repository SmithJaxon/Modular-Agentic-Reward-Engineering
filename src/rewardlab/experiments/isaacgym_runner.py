"""
Summary: Minimal actual-backend Isaac experiment runner for RewardLab candidate evaluation.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from numbers import Real
from typing import Any

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.experiments.execution_service import (
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
)
from rewardlab.experiments.reward_program import RewardProgram


class IsaacGymExperimentRunner:
    """Execute one Isaac-style rollout and score it with a reward program."""

    def __init__(
        self,
        *,
        backend: IsaacGymBackend | None = None,
        default_max_episode_steps: int = 200,
    ) -> None:
        """Store the backend adapter and default rollout budget."""

        self.backend = backend or IsaacGymBackend()
        self.default_max_episode_steps = default_max_episode_steps

    def __call__(
        self,
        execution_request: ExecutionRequest,
        reward_program: RewardProgram,
    ) -> ExecutionOutcome:
        """Execute a single Isaac rollout for the supplied reward program."""

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
            observation, _ = _normalize_reset_output(environment.reset(seed=execution_request.seed))
            episode_reward = 0.0
            environment_reward = 0.0
            terminated = False
            truncated = False
            event_trace: list[dict[str, Any]] = []

            for step_index in range(1, step_limit + 1):
                action = _select_default_action(environment, observation)
                next_observation, env_reward, terminated, truncated, info = _normalize_step_output(
                    environment.step(action)
                )
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


def _normalize_reset_output(reset_output: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize backend reset output to `(observation, info)` form."""

    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        observation, info = reset_output
        return observation, _normalize_info(info)
    return reset_output, {}


def _normalize_step_output(
    step_output: Any,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Normalize backend step output to Gymnasium-like rollout semantics."""

    if not isinstance(step_output, tuple):
        raise ExecutionError("Isaac environment step must return a tuple result")
    if len(step_output) == 5:
        observation, reward, terminated, truncated, info = step_output
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            _normalize_info(info),
        )
    if len(step_output) == 4:
        observation, reward, terminated, info = step_output
        return observation, float(reward), bool(terminated), False, _normalize_info(info)
    raise ExecutionError("Isaac environment step must return 4 or 5 values")


def _select_default_action(environment: Any, observation: Any) -> Any:
    """Choose a deterministic default action for an Isaac-style environment."""

    for method_name in ("default_action", "zero_action"):
        action_factory = getattr(environment, method_name, None)
        if callable(action_factory):
            return action_factory()

    action_space = getattr(environment, "action_space", None)
    if action_space is not None:
        discrete_size = getattr(action_space, "n", None)
        if isinstance(discrete_size, int) and discrete_size > 0:
            return 0

        shape = getattr(action_space, "shape", None)
        if isinstance(shape, Sequence):
            return _zeros_from_shape(shape)

    for attribute_name in ("num_actions", "action_dim", "action_dimensions"):
        size = getattr(environment, attribute_name, None)
        if isinstance(size, int) and size > 0:
            return [0.0] * size

    if _is_numeric_sequence(observation, min_length=1):
        return [0.0]
    return 0.0


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
    """Evaluate the reward program against a single Isaac transition."""

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
            f"reward entrypoint requires unsupported parameters for Isaac rollout: {joined}"
        )
    return kwargs


def _normalize_info(info: Any) -> dict[str, Any]:
    """Normalize backend info payloads to a dictionary."""

    if isinstance(info, dict):
        return dict(info)
    return {"raw_info": info}


def _zeros_from_shape(shape: Sequence[Any]) -> Any:
    """Construct a nested zero structure matching a simple shape tuple."""

    normalized = [int(item) for item in shape]
    if not normalized:
        return 0.0
    if len(normalized) == 1:
        return [0.0] * max(normalized[0], 1)
    return [_zeros_from_shape(normalized[1:]) for _ in range(max(normalized[0], 1))]


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
