"""
Summary: Real Gymnasium PPO execution helpers for reward-program experiments.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from rewardlab.experiments.backends.base import ExperimentInput

_DEFAULT_DATA_DIR = ".rewardlab"
_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

try:
    import gymnasium_robotics  # noqa: F401
except Exception:  # noqa: BLE001
    gymnasium_robotics = None  # type: ignore[assignment]


@dataclass(slots=True, frozen=True)
class GymnasiumExecutionConfig:
    """
    Carry resolved runtime settings for Gymnasium execution.
    """

    execution_mode: str = "deterministic"
    llm_provider: str = "none"
    llm_model: str = "gpt-4o-mini"
    ppo_total_timesteps: int = 4096
    ppo_num_envs: int = 4
    ppo_n_steps: int = 256
    ppo_batch_size: int = 128
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_n_epochs: int = 10
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_activation_fn: str = "tanh"
    ppo_policy_hidden_sizes: tuple[int, ...] = (64, 64)
    evaluation_episodes: int = 5
    reflection_episodes: int = 2
    reflection_interval_steps: int = 1024
    train_seed: int = 7
    robustness_budget_scale: float = 0.5
    human_feedback_enabled: bool = False
    peer_feedback_enabled: bool = False

    @property
    def is_real_execution(self) -> bool:
        """
        Report whether PPO execution should be used.
        """
        return self.execution_mode == "ppo"

    @classmethod
    def from_overrides(
        cls,
        overrides: dict[str, Any],
        *,
        variant_label: str,
    ) -> GymnasiumExecutionConfig:
        """
        Normalize flat session metadata and probe overrides into runtime settings.
        """
        planned_timesteps = _optional_int_value(
            overrides,
            "planned_ppo_total_timesteps",
            minimum=64,
        )
        planned_evaluation_episodes = _optional_int_value(
            overrides,
            "planned_evaluation_episodes",
            minimum=1,
        )
        planned_reflection_episodes = _optional_int_value(
            overrides,
            "planned_reflection_episodes",
            minimum=0,
        )
        planned_reflection_interval = _optional_int_value(
            overrides,
            "planned_reflection_interval_steps",
            minimum=64,
        )
        planned_probe_scale = _optional_float_value(
            overrides,
            "planned_robustness_budget_scale",
            minimum=0.1,
        )
        config = cls(
            execution_mode=_string_value(overrides, "execution_mode", "deterministic"),
            llm_provider=_string_value(overrides, "llm_provider", "none"),
            llm_model=_string_value(overrides, "llm_model", "gpt-4o-mini"),
            ppo_total_timesteps=planned_timesteps
            or _int_value(overrides, "ppo_total_timesteps", 4096, minimum=64),
            ppo_num_envs=_int_value(overrides, "ppo_num_envs", 4, minimum=1),
            ppo_n_steps=_int_value(overrides, "ppo_n_steps", 256, minimum=32),
            ppo_batch_size=_int_value(overrides, "ppo_batch_size", 128, minimum=32),
            ppo_learning_rate=_float_value(
                overrides,
                "ppo_learning_rate",
                3e-4,
                minimum=1e-6,
            ),
            ppo_gamma=_float_value(
                overrides,
                "ppo_gamma",
                0.99,
                minimum=1e-6,
                maximum=1.0,
            ),
            ppo_gae_lambda=_float_value(
                overrides,
                "ppo_gae_lambda",
                0.95,
                minimum=1e-6,
                maximum=1.0,
            ),
            ppo_clip_range=_float_value(
                overrides,
                "ppo_clip_range",
                0.2,
                minimum=1e-6,
            ),
            ppo_n_epochs=_int_value(overrides, "ppo_n_epochs", 10, minimum=1),
            ppo_ent_coef=_float_value(overrides, "ppo_ent_coef", 0.0, minimum=0.0),
            ppo_vf_coef=_float_value(overrides, "ppo_vf_coef", 0.5, minimum=0.0),
            ppo_max_grad_norm=_float_value(
                overrides,
                "ppo_max_grad_norm",
                0.5,
                minimum=1e-6,
            ),
            ppo_activation_fn=_string_value(overrides, "ppo_activation_fn", "tanh"),
            ppo_policy_hidden_sizes=_int_tuple_value(
                overrides,
                "ppo_policy_hidden_sizes",
                default=(64, 64),
                minimum=1,
            ),
            evaluation_episodes=planned_evaluation_episodes
            or _int_value(overrides, "evaluation_episodes", 5, minimum=1),
            reflection_episodes=planned_reflection_episodes
            if planned_reflection_episodes is not None
            else _int_value(overrides, "reflection_episodes", 2, minimum=0),
            reflection_interval_steps=planned_reflection_interval
            or _int_value(
                overrides,
                "reflection_interval_steps",
                1024,
                minimum=64,
            ),
            train_seed=_int_value(overrides, "train_seed", 7),
            robustness_budget_scale=planned_probe_scale
            or _float_value(
                overrides,
                "robustness_budget_scale",
                0.5,
                minimum=0.1,
            ),
            human_feedback_enabled=_bool_value(overrides, "human_feedback_enabled", False),
            peer_feedback_enabled=_bool_value(overrides, "peer_feedback_enabled", False),
        )
        if variant_label == "default":
            return config

        scaled_timesteps = max(
            64,
            int(round(config.ppo_total_timesteps * config.robustness_budget_scale)),
        )
        scaled_reflection_interval = max(
            64,
            int(round(config.reflection_interval_steps * config.robustness_budget_scale)),
        )
        scaled_evaluation_episodes = max(
            1,
            int(round(config.evaluation_episodes * config.robustness_budget_scale)),
        )
        scaled_reflection_episodes = max(
            0,
            int(round(config.reflection_episodes * config.robustness_budget_scale)),
        )
        return cls(
            execution_mode=config.execution_mode,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            ppo_total_timesteps=scaled_timesteps,
            ppo_num_envs=config.ppo_num_envs,
            ppo_n_steps=config.ppo_n_steps,
            ppo_batch_size=config.ppo_batch_size,
            ppo_learning_rate=config.ppo_learning_rate,
            ppo_gamma=config.ppo_gamma,
            ppo_gae_lambda=config.ppo_gae_lambda,
            ppo_clip_range=config.ppo_clip_range,
            ppo_n_epochs=config.ppo_n_epochs,
            ppo_ent_coef=config.ppo_ent_coef,
            ppo_vf_coef=config.ppo_vf_coef,
            ppo_max_grad_norm=config.ppo_max_grad_norm,
            ppo_activation_fn=config.ppo_activation_fn,
            ppo_policy_hidden_sizes=config.ppo_policy_hidden_sizes,
            evaluation_episodes=scaled_evaluation_episodes,
            reflection_episodes=scaled_reflection_episodes,
            reflection_interval_steps=scaled_reflection_interval,
            train_seed=config.train_seed,
            robustness_budget_scale=config.robustness_budget_scale,
            human_feedback_enabled=config.human_feedback_enabled,
            peer_feedback_enabled=config.peer_feedback_enabled,
        )


@dataclass(slots=True, frozen=True)
class RewardComputation:
    """
    Hold one reward-program output.
    """

    reward: float
    components: dict[str, float]


@dataclass(slots=True, frozen=True)
class EvaluationSnapshot:
    """
    Capture one checkpointed evaluation summary.
    """

    timesteps: int
    env_return_mean: float
    shaped_return_mean: float
    episode_length_mean: float
    component_means: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        """
        Convert the snapshot into a JSON-safe dictionary.
        """
        return {
            "timesteps": self.timesteps,
            "env_return_mean": self.env_return_mean,
            "shaped_return_mean": self.shaped_return_mean,
            "episode_length_mean": self.episode_length_mean,
            "component_means": self.component_means,
        }


@dataclass(slots=True, frozen=True)
class GymnasiumExecutionArtifacts:
    """
    Track persisted runtime artifacts for one execution.
    """

    artifact_dir: Path
    reward_program_path: Path
    summary_path: Path
    reflection_path: Path
    model_path: Path

    def refs(self) -> tuple[str, ...]:
        """
        Expose artifact paths in stable string form.
        """
        return (
            str(self.reward_program_path),
            str(self.summary_path),
            str(self.reflection_path),
            str(self.model_path),
        )


@dataclass(slots=True, frozen=True)
class GymnasiumRunResult:
    """
    Bundle primary execution metrics, reflection text, and artifacts.
    """

    score: float
    metrics: dict[str, Any]
    performance_summary: str
    reflection_summary: str
    artifacts: GymnasiumExecutionArtifacts


class RewardProgramError(RuntimeError):
    """
    Represent invalid reward-program source or execution failures.
    """


class RewardProgram:
    """
    Compile and execute a reward function against Gymnasium transitions.
    """

    def __init__(self, source: str) -> None:
        """
        Normalize the source and resolve the required `compute_reward` function.
        """
        self.source = normalize_reward_program_source(source)
        self._function = self._compile(self.source)

    @staticmethod
    def _compile(source: str) -> Any:
        """
        Compile reward source into a callable `compute_reward` function.
        """
        safe_builtins = {
            "__import__": __import__,
            "abs": abs,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "rewardlab_reward_program",
            "math": math,
            "np": np,
            "numpy": np,
        }
        try:
            exec(source, namespace, namespace)
        except Exception as exc:  # pragma: no cover - exercised through loader callers.
            raise RewardProgramError(
                f"reward program compilation failed ({exc.__class__.__name__})"
            ) from exc
        compute_reward = namespace.get("compute_reward")
        if not callable(compute_reward):
            raise RewardProgramError("reward program must define callable compute_reward")
        return compute_reward

    def evaluate(
        self,
        observation: np.ndarray[Any, Any],
        action: np.ndarray[Any, Any],
        next_observation: np.ndarray[Any, Any],
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> RewardComputation:
        """
        Execute the reward function on one transition and normalize its result.
        """
        try:
            raw_result = self._function(
                observation,
                action,
                next_observation,
                float(env_reward),
                bool(terminated),
                bool(truncated),
                dict(info),
            )
        except Exception as exc:
            raise RewardProgramError(
                f"reward program execution failed ({exc.__class__.__name__})"
            ) from exc
        if not isinstance(raw_result, tuple) or len(raw_result) != 2:
            raise RewardProgramError(
                "reward program must return a tuple of (reward, component_dict)"
            )
        raw_reward, raw_components = raw_result
        try:
            reward = float(raw_reward)
        except (TypeError, ValueError) as exc:
            raise RewardProgramError("reward program returned a non-numeric reward") from exc
        if not isinstance(raw_components, dict):
            raise RewardProgramError("reward program components must be returned as a dictionary")
        components: dict[str, float] = {}
        for key, value in raw_components.items():
            try:
                components[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise RewardProgramError(
                    f"reward component {key!r} must be numeric"
                ) from exc
        return RewardComputation(reward=reward, components=components)


class RewardShapingWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """
    Replace the environment reward with a compiled reward program and track episode statistics.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        reward_program: RewardProgram,
        *,
        seed: int,
        overrides: dict[str, Any],
    ) -> None:
        """
        Initialize reward shaping and probe-specific perturbations for one environment instance.
        """
        super().__init__(env)
        self._reward_program = reward_program
        self._rng = np.random.default_rng(seed)
        self._dropout_rate = _float_value(overrides, "dropout_rate", 0.0, minimum=0.0)
        self._delay_steps = _int_value(overrides, "delay_steps", 0, minimum=0)
        self._reward_queue: deque[float] = deque()
        self._last_observation = np.zeros(1, dtype=np.float32)
        self._completed_episode_returns: list[float] = []
        self._completed_shaped_returns: list[float] = []
        self._completed_episode_lengths: list[int] = []
        self._component_sums: dict[str, float] = defaultdict(float)
        self._component_counts: dict[str, int] = defaultdict(int)
        self._current_env_return = 0.0
        self._current_shaped_return = 0.0
        self._current_episode_length = 0
        self._apply_probe_overrides(overrides)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """
        Reset the wrapped environment and clear in-progress delayed reward state.
        """
        observation, info = self.env.reset(seed=seed, options=options)
        self._reward_queue.clear()
        self._current_env_return = 0.0
        self._current_shaped_return = 0.0
        self._current_episode_length = 0
        self._last_observation = self._mask_observation(np.asarray(observation, dtype=np.float32))
        return self._last_observation.copy(), dict(info)

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        """
        Step the wrapped environment and substitute the reward-program output.
        """
        next_observation_raw, env_reward, terminated, truncated, info = self.env.step(action)
        next_observation = self._mask_observation(
            np.asarray(next_observation_raw, dtype=np.float32)
        )
        action_array = np.asarray(action, dtype=np.float32)
        computation = self._reward_program.evaluate(
            observation=self._last_observation,
            action=action_array,
            next_observation=next_observation,
            env_reward=float(env_reward),
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )
        delivered_reward = self._delay_reward(
            computation.reward,
            terminal=terminated or truncated,
        )
        self._current_env_return += float(env_reward)
        self._current_shaped_return += delivered_reward
        self._current_episode_length += 1
        for name, value in computation.components.items():
            self._component_sums[name] += value
            self._component_counts[name] += 1
        if terminated or truncated:
            self._completed_episode_returns.append(self._current_env_return)
            self._completed_shaped_returns.append(self._current_shaped_return)
            self._completed_episode_lengths.append(self._current_episode_length)
        info_payload = dict(info)
        info_payload["rewardlab"] = {
            "env_reward": float(env_reward),
            "shaped_reward": computation.reward,
            "components": computation.components,
        }
        self._last_observation = next_observation
        return next_observation.copy(), delivered_reward, terminated, truncated, info_payload

    def statistics_snapshot(self, *, clear: bool) -> EvaluationSnapshot:
        """
        Build a mean summary for completed episodes and observed reward components.
        """
        component_means = {
            name: self._component_sums[name] / max(self._component_counts[name], 1)
            for name in sorted(self._component_sums)
        }
        snapshot = EvaluationSnapshot(
            timesteps=0,
            env_return_mean=_mean(self._completed_episode_returns),
            shaped_return_mean=_mean(self._completed_shaped_returns),
            episode_length_mean=_mean(self._completed_episode_lengths),
            component_means=component_means,
        )
        if clear:
            self._completed_episode_returns.clear()
            self._completed_shaped_returns.clear()
            self._completed_episode_lengths.clear()
            self._component_sums.clear()
            self._component_counts.clear()
        return snapshot

    def _mask_observation(
        self,
        observation: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        Apply observation-dropout perturbation when configured.
        """
        if self._dropout_rate <= 0.0:
            return observation
        mask = self._rng.random(observation.shape) >= self._dropout_rate
        masked = (observation * mask).astype(observation.dtype, copy=False)
        return cast(np.ndarray[Any, Any], masked)

    def _delay_reward(self, reward: float, *, terminal: bool) -> float:
        """
        Delay reward delivery for probe variants that model sparse or lagged feedback.
        """
        if self._delay_steps <= 0:
            return reward
        self._reward_queue.append(reward)
        delivered = 0.0
        if len(self._reward_queue) > self._delay_steps:
            delivered = self._reward_queue.popleft()
        if terminal:
            while self._reward_queue:
                delivered += self._reward_queue.popleft()
        return delivered

    def _apply_probe_overrides(self, overrides: dict[str, Any]) -> None:
        """
        Apply environment perturbations used by robustness probes when supported.
        """
        gravity_scale = _float_value(overrides, "gravity_scale", 1.0, minimum=0.1)
        if gravity_scale == 1.0:
            return

        if hasattr(self.env.unwrapped, "gravity"):
            gravity_value = self.env.unwrapped.gravity
            if isinstance(gravity_value, int | float):
                self.env.unwrapped.gravity = float(gravity_value) * gravity_scale
                return

        model = getattr(self.env.unwrapped, "model", None)
        opt = getattr(model, "opt", None)
        gravity = getattr(opt, "gravity", None)
        if gravity is None:
            return
        gravity_array = np.asarray(gravity, dtype=np.float64)
        scaled = gravity_array * gravity_scale
        if hasattr(gravity, "__setitem__"):
            gravity[...] = scaled


def normalize_reward_program_source(source: str) -> str:
    """
    Normalize raw or fenced reward-program text into executable Python source.
    """
    trimmed = source.strip()
    if not trimmed:
        raise RewardProgramError("reward program source is empty")
    match = _CODE_BLOCK_PATTERN.search(trimmed)
    if match is not None:
        return match.group(1).strip()
    return trimmed


def default_env_reward_program_source() -> str:
    """
    Return a baseline program that mirrors the environment reward.
    """
    return (
        "def compute_reward(observation, action, next_observation, env_reward, "
        "terminated, truncated, info):\n"
        "    reward = float(env_reward)\n"
        "    return reward, {\"env_reward\": reward}\n"
    )


def run_gymnasium_ppo_experiment(payload: ExperimentInput) -> GymnasiumRunResult:
    """
    Train and evaluate a PPO policy under the provided reward program.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed; install optional dependency with "
            "'pip install -e .[rl]' after adding stable-baselines3 to the environment"
        ) from exc

    config = GymnasiumExecutionConfig.from_overrides(
        payload.overrides,
        variant_label=payload.variant_label,
    )
    reward_program = RewardProgram(payload.reward_definition)
    rollout_size = config.ppo_num_envs * config.ppo_n_steps
    total_timesteps = _round_up(config.ppo_total_timesteps, rollout_size)
    reflection_interval = _round_up(
        max(config.reflection_interval_steps, rollout_size),
        rollout_size,
    )
    batch_size = min(config.ppo_batch_size, rollout_size)
    activation_fn = _resolve_activation_fn(config.ppo_activation_fn)
    policy_kwargs = {
        "activation_fn": activation_fn,
        "net_arch": list(config.ppo_policy_hidden_sizes),
    }

    artifacts = _prepare_artifacts(payload, reward_program.source)
    train_env = DummyVecEnv(
        [
            _make_env_factory(
                payload=payload,
                reward_program=reward_program,
                seed=config.train_seed + index,
            )
            for index in range(config.ppo_num_envs)
        ]
    )
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.ppo_learning_rate,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        n_epochs=config.ppo_n_epochs,
        ent_coef=config.ppo_ent_coef,
        vf_coef=config.ppo_vf_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        n_steps=config.ppo_n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config.train_seed,
        device="auto",
    )

    snapshots: list[EvaluationSnapshot] = []
    trained_timesteps = 0
    while trained_timesteps < total_timesteps:
        learn_timesteps = min(reflection_interval, total_timesteps - trained_timesteps)
        model.learn(total_timesteps=learn_timesteps, reset_num_timesteps=False, progress_bar=False)
        trained_timesteps += learn_timesteps
        if config.reflection_episodes > 0:
            snapshots.append(
                _evaluate_model(
                    model=model,
                    payload=payload,
                    reward_program=reward_program,
                    episodes=config.reflection_episodes,
                    seed=config.train_seed + 1000 + trained_timesteps,
                    timesteps=trained_timesteps,
                )
            )

    final_snapshot = _evaluate_model(
        model=model,
        payload=payload,
        reward_program=reward_program,
        episodes=config.evaluation_episodes,
        seed=config.train_seed + 5000,
        timesteps=trained_timesteps,
    )
    model.save(str(artifacts.model_path))
    train_env.close()

    reflection_summary = build_reflection_summary(
        environment_id=payload.environment_id,
        checkpoints=snapshots,
        final_snapshot=final_snapshot,
        reflection_interval_steps=reflection_interval,
    )
    evaluation_episodes_consumed = config.evaluation_episodes + (
        len(snapshots) * config.reflection_episodes
    )
    metrics = {
        "backend": "gymnasium",
        "variant_label": payload.variant_label,
        "seed": payload.seed,
        "overrides": payload.overrides,
        "runtime_available": True,
        "execution_mode": config.execution_mode,
        "llm_provider": config.llm_provider,
        "score": final_snapshot.env_return_mean,
        "env_return_mean": final_snapshot.env_return_mean,
        "shaped_return_mean": final_snapshot.shaped_return_mean,
        "episode_length_mean": final_snapshot.episode_length_mean,
        "component_means": final_snapshot.component_means,
        "reflection_checkpoints": [snapshot.to_json() for snapshot in snapshots],
        "reflection_checkpoint_count": len(snapshots),
        "ppo_policy_hidden_sizes": list(config.ppo_policy_hidden_sizes),
        "ppo_activation_fn": config.ppo_activation_fn,
        "ppo_gamma": config.ppo_gamma,
        "ppo_gae_lambda": config.ppo_gae_lambda,
        "ppo_clip_range": config.ppo_clip_range,
        "ppo_n_epochs": config.ppo_n_epochs,
        "ppo_ent_coef": config.ppo_ent_coef,
        "ppo_vf_coef": config.ppo_vf_coef,
        "ppo_max_grad_norm": config.ppo_max_grad_norm,
        "final_evaluation_episodes_consumed": config.evaluation_episodes,
        "reflection_evaluation_episodes_consumed": len(snapshots) * config.reflection_episodes,
        "evaluation_episodes_consumed": evaluation_episodes_consumed,
        "total_timesteps": trained_timesteps,
    }
    performance_summary = (
        "gymnasium performance "
        f"variant={payload.variant_label} iteration={payload.iteration_index} "
        f"env_return={final_snapshot.env_return_mean:.3f} "
        f"shaped_return={final_snapshot.shaped_return_mean:.3f} "
        f"episode_length={final_snapshot.episode_length_mean:.1f}"
    )
    artifacts.summary_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "performance_summary": performance_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts.reflection_path.write_text(reflection_summary, encoding="utf-8")
    return GymnasiumRunResult(
        score=final_snapshot.env_return_mean,
        metrics=metrics,
        performance_summary=performance_summary,
        reflection_summary=reflection_summary,
        artifacts=artifacts,
    )


def build_reflection_summary(
    *,
    environment_id: str,
    checkpoints: list[EvaluationSnapshot],
    final_snapshot: EvaluationSnapshot,
    reflection_interval_steps: int,
) -> str:
    """
    Convert PPO checkpoint statistics into a Eureka-style reflection summary.
    """
    lines = [
        "We trained a PPO policy using the provided reward function code and tracked reward "
        "components together with task fitness checkpoints.",
        f"Environment: {environment_id}",
        f"Checkpoint interval (timesteps): {reflection_interval_steps}",
        _series_line("env_return", [snapshot.env_return_mean for snapshot in checkpoints]),
        _series_line("shaped_return", [snapshot.shaped_return_mean for snapshot in checkpoints]),
        _series_line("episode_length", [snapshot.episode_length_mean for snapshot in checkpoints]),
    ]
    component_names = {
        name for snapshot in checkpoints for name in snapshot.component_means
    } | set(final_snapshot.component_means)
    for name in sorted(component_names):
        lines.append(
            _series_line(
                name,
                [snapshot.component_means.get(name, 0.0) for snapshot in checkpoints],
            )
        )
    lines.extend(
        [
            "Final evaluation snapshot:",
            f"env_return_mean={final_snapshot.env_return_mean:.4f}",
            f"shaped_return_mean={final_snapshot.shaped_return_mean:.4f}",
            f"episode_length_mean={final_snapshot.episode_length_mean:.4f}",
            "Use these signals to improve the reward: rewrite weak components with near-constant "
            "values, rescale oversized terms, and favor changes that increase environment return "
            "without shortening episodes.",
        ]
    )
    return "\n".join(lines)


def _make_env_factory(
    *,
    payload: ExperimentInput,
    reward_program: RewardProgram,
    seed: int,
) -> Any:
    """
    Build one delayed environment factory for SB3 vectorized training.
    """

    def _factory() -> RewardShapingWrapper:
        env = gym.make(payload.environment_id)
        env.reset(seed=seed)
        return RewardShapingWrapper(
            env=env,
            reward_program=reward_program,
            seed=seed,
            overrides=payload.overrides,
        )

    return _factory


def _evaluate_model(
    *,
    model: Any,
    payload: ExperimentInput,
    reward_program: RewardProgram,
    episodes: int,
    seed: int,
    timesteps: int,
) -> EvaluationSnapshot:
    """
    Evaluate the learned policy on a single wrapped environment and summarize the results.
    """
    env = RewardShapingWrapper(
        env=gym.make(payload.environment_id),
        reward_program=reward_program,
        seed=seed,
        overrides=payload.overrides,
    )
    observation, _ = env.reset(seed=seed)
    completed_episodes = 0
    while completed_episodes < episodes:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            completed_episodes += 1
            if completed_episodes < episodes:
                observation, _ = env.reset(seed=seed + completed_episodes)
    summary = env.statistics_snapshot(clear=True)
    env.close()  # type: ignore[no-untyped-call]
    return EvaluationSnapshot(
        timesteps=timesteps,
        env_return_mean=summary.env_return_mean,
        shaped_return_mean=summary.shaped_return_mean,
        episode_length_mean=summary.episode_length_mean,
        component_means=summary.component_means,
    )


def _prepare_artifacts(
    payload: ExperimentInput,
    reward_program_source: str,
) -> GymnasiumExecutionArtifacts:
    """
    Prepare the artifact directory for one Gymnasium execution.
    """
    artifact_dir = (
        _data_dir(payload.overrides)
        / "experiments"
        / payload.environment_backend.value
        / payload.session_id
        / f"iter-{payload.iteration_index:03d}"
        / payload.variant_label
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reward_program_path = artifact_dir / "reward_program.py"
    reward_program_path.write_text(reward_program_source, encoding="utf-8")
    summary_path = artifact_dir / "summary.json"
    reflection_path = artifact_dir / "reflection.txt"
    model_path = artifact_dir / "policy_model.zip"
    return GymnasiumExecutionArtifacts(
        artifact_dir=artifact_dir,
        reward_program_path=reward_program_path,
        summary_path=summary_path,
        reflection_path=reflection_path,
        model_path=model_path,
    )


def _data_dir(overrides: dict[str, Any]) -> Path:
    """
    Resolve the artifact root from the shared data-dir convention.
    """
    artifact_root = overrides.get("artifact_root")
    if artifact_root is not None:
        return Path(str(artifact_root))
    return Path(os.getenv("REWARDLAB_DATA_DIR", _DEFAULT_DATA_DIR))


def _mean(values: Sequence[float | int]) -> float:
    """
    Return the arithmetic mean for a possibly empty list.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _series_line(name: str, values: list[float]) -> str:
    """
    Format one checkpoint metric series for reflection prompts.
    """
    if not values:
        return f"{name}: []"
    formatted = [f"{value:.3f}" for value in values]
    return (
        f"{name}: {formatted}, Max: {max(values):.3f}, "
        f"Mean: {_mean(values):.3f}, Min: {min(values):.3f}"
    )


def _round_up(value: int, multiple: int) -> int:
    """
    Round a positive integer up to the nearest multiple.
    """
    return multiple * math.ceil(value / multiple)


def _string_value(overrides: dict[str, Any], key: str, default: str) -> str:
    """
    Read a normalized string value from overrides.
    """
    value = overrides.get(key, default)
    text = str(value).strip().lower()
    return text or default


def _int_value(
    overrides: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """
    Read an integer value from overrides with optional lower bound enforcement.
    """
    value = overrides.get(key, default)
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    if minimum is not None:
        return max(minimum, normalized)
    return normalized


def _optional_int_value(
    overrides: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    """
    Read an optional integer override, preserving absent values as None.
    """
    if key not in overrides or overrides.get(key) in {None, ""}:
        return None
    return _int_value(overrides, key, int(overrides[key]), minimum=minimum)


def _float_value(
    overrides: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Read a float value from overrides with optional lower bound enforcement.
    """
    value = overrides.get(key, default)
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = default
    if minimum is not None:
        normalized = max(minimum, normalized)
    if maximum is not None:
        normalized = min(maximum, normalized)
    return normalized


def _optional_float_value(
    overrides: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
) -> float | None:
    """
    Read an optional float override, preserving absent values as None.
    """
    if key not in overrides or overrides.get(key) in {None, ""}:
        return None
    return _float_value(overrides, key, float(overrides[key]), minimum=minimum)


def _int_tuple_value(
    overrides: dict[str, Any],
    key: str,
    *,
    default: tuple[int, ...],
    minimum: int = 1,
) -> tuple[int, ...]:
    """
    Read a tuple of positive integers from overrides.
    """
    raw = overrides.get(key, default)
    if isinstance(raw, tuple | list):
        values: list[int] = []
        for item in raw:
            if isinstance(item, int | float):
                values.append(max(minimum, int(item)))
        if values:
            return tuple(values)
    return default


def _resolve_activation_fn(name: str) -> Any:
    """
    Resolve activation-function name into a torch.nn module class for SB3.
    """
    try:
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required to resolve policy activation function") from exc
    normalized = name.strip().lower()
    mapping = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
    }
    return mapping.get(normalized, nn.Tanh)


def _bool_value(overrides: dict[str, Any], key: str, default: bool) -> bool:
    """
    Read a boolean value from overrides, accepting common string forms.
    """
    value = overrides.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
