"""
Summary: Session-level adaptive budget planning for PPO-backed Gymnasium experiments.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.schemas.session_config import EnvironmentBackend


@dataclass(slots=True, frozen=True)
class AdaptiveBudgetSettings:
    """
    Hold normalized session-level budget settings for adaptive PPO execution.
    """

    budget_mode: str
    total_training_timesteps: int
    total_evaluation_episodes: int
    max_llm_calls: int
    candidate_train_floor_timesteps: int
    candidate_train_ceiling_timesteps: int | None
    target_reflection_checkpoints: int
    ppo_num_envs: int
    ppo_n_steps: int
    evaluation_episode_ceiling: int
    reflection_episode_ceiling: int
    robustness_budget_scale: float
    variant_count: int

    @property
    def rollout_size(self) -> int:
        """
        Return the PPO rollout size implied by the vectorized environment settings.
        """
        return self.ppo_num_envs * self.ppo_n_steps

    @property
    def is_adaptive(self) -> bool:
        """
        Report whether adaptive planning is enabled.
        """
        return self.budget_mode == "adaptive"

    @classmethod
    def from_session(
        cls,
        session: dict[str, Any],
        metadata: dict[str, Any],
    ) -> AdaptiveBudgetSettings:
        """
        Build normalized budget settings from session rows and metadata.
        """
        rollout_num_envs = _int_value(metadata, "ppo_num_envs", 4, minimum=1)
        rollout_n_steps = _int_value(metadata, "ppo_n_steps", 256, minimum=32)
        rollout_size = rollout_num_envs * rollout_n_steps
        probe_scale = _float_value(metadata, "robustness_budget_scale", 0.5, minimum=0.1)
        variant_count = _variant_count(EnvironmentBackend(session["environment_backend"]))
        legacy_primary_timesteps = _int_value(metadata, "ppo_total_timesteps", 4096, minimum=64)
        legacy_reflection_interval = _int_value(
            metadata,
            "reflection_interval_steps",
            1024,
            minimum=64,
        )
        legacy_checkpoint_count = max(
            1,
            math.ceil(legacy_primary_timesteps / max(legacy_reflection_interval, rollout_size)),
        )
        legacy_final_eval = _int_value(metadata, "evaluation_episodes", 5, minimum=1)
        legacy_reflection_eval = _int_value(metadata, "reflection_episodes", 2, minimum=1)
        per_candidate_eval = legacy_final_eval + (legacy_checkpoint_count * legacy_reflection_eval)
        default_total_train = int(
            round(
                legacy_primary_timesteps
                * session["max_iterations"]
                * (1.0 + (variant_count * probe_scale))
            )
        )
        default_total_eval = int(
            round(
                per_candidate_eval
                * session["max_iterations"]
                * (1.0 + (variant_count * probe_scale))
            )
        )
        llm_provider = _string_value(metadata, "llm_provider", "none")
        default_llm_calls = max(0, session["max_iterations"] - 1) if llm_provider == "openai" else 0
        return cls(
            budget_mode=_string_value(metadata, "budget_mode", "adaptive"),
            total_training_timesteps=_int_value(
                metadata,
                "total_training_timesteps",
                default_total_train,
                minimum=rollout_size,
            ),
            total_evaluation_episodes=_int_value(
                metadata,
                "total_evaluation_episodes",
                max(1, default_total_eval),
                minimum=1,
            ),
            max_llm_calls=_int_value(metadata, "max_llm_calls", default_llm_calls, minimum=0),
            candidate_train_floor_timesteps=_int_value(
                metadata,
                "candidate_train_floor_timesteps",
                rollout_size,
                minimum=rollout_size,
            ),
            candidate_train_ceiling_timesteps=_optional_int_value(
                metadata,
                "candidate_train_ceiling_timesteps",
                minimum=rollout_size,
            ),
            target_reflection_checkpoints=_int_value(
                metadata,
                "target_reflection_checkpoints",
                4,
                minimum=1,
            ),
            ppo_num_envs=rollout_num_envs,
            ppo_n_steps=rollout_n_steps,
            evaluation_episode_ceiling=_int_value(
                metadata,
                "evaluation_episodes",
                5,
                minimum=1,
            ),
            reflection_episode_ceiling=_int_value(
                metadata,
                "reflection_episodes",
                2,
                minimum=1,
            ),
            robustness_budget_scale=probe_scale,
            variant_count=variant_count,
        )


@dataclass(slots=True, frozen=True)
class AdaptiveBudgetPlan:
    """
    Capture the per-iteration resource plan selected from the session budget.
    """

    primary_train_timesteps: int
    evaluation_episodes: int
    reflection_episodes: int
    reflection_interval_steps: int
    robustness_budget_scale: float
    rationale: str

    def as_metadata_overrides(self) -> dict[str, int | float | str]:
        """
        Convert the plan into metadata overrides consumed by the PPO runtime.
        """
        return {
            "planned_ppo_total_timesteps": self.primary_train_timesteps,
            "planned_evaluation_episodes": self.evaluation_episodes,
            "planned_reflection_episodes": self.reflection_episodes,
            "planned_reflection_interval_steps": self.reflection_interval_steps,
            "planned_robustness_budget_scale": self.robustness_budget_scale,
            "planned_budget_rationale": self.rationale,
        }


def initialize_budget_metadata(
    session: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Seed session metadata with concrete adaptive budget totals and spend counters.
    """
    if EnvironmentBackend(session["environment_backend"]) is not EnvironmentBackend.GYMNASIUM:
        return metadata
    if _string_value(metadata, "execution_mode", "deterministic") != "ppo":
        return metadata

    settings = AdaptiveBudgetSettings.from_session(session, metadata)
    updated = dict(metadata)
    updated.setdefault("budget_mode", settings.budget_mode)
    updated.setdefault("total_training_timesteps", settings.total_training_timesteps)
    updated.setdefault("total_evaluation_episodes", settings.total_evaluation_episodes)
    updated.setdefault("max_llm_calls", settings.max_llm_calls)
    updated.setdefault(
        "candidate_train_floor_timesteps",
        settings.candidate_train_floor_timesteps,
    )
    if settings.candidate_train_ceiling_timesteps is not None:
        updated.setdefault(
            "candidate_train_ceiling_timesteps",
            settings.candidate_train_ceiling_timesteps,
        )
    updated.setdefault(
        "target_reflection_checkpoints",
        settings.target_reflection_checkpoints,
    )
    updated.setdefault("spent_training_timesteps", 0)
    updated.setdefault("spent_evaluation_episodes", 0)
    updated.setdefault("llm_calls_used", 0)
    updated.setdefault("budget_history", [])
    return updated


def plan_iteration_budget(
    session: dict[str, Any],
    metadata: dict[str, Any],
    iteration_index: int,
) -> AdaptiveBudgetPlan | None:
    """
    Plan the next PPO iteration from the remaining session budget.
    """
    if EnvironmentBackend(session["environment_backend"]) is not EnvironmentBackend.GYMNASIUM:
        return None
    if _string_value(metadata, "execution_mode", "deterministic") != "ppo":
        return None

    settings = AdaptiveBudgetSettings.from_session(session, metadata)
    if not settings.is_adaptive:
        return None

    remaining_iterations = max(1, session["max_iterations"] - iteration_index)
    remaining_train = max(
        0,
        settings.total_training_timesteps - _int_value(metadata, "spent_training_timesteps", 0),
    )
    remaining_eval = max(
        0,
        settings.total_evaluation_episodes
        - _int_value(metadata, "spent_evaluation_episodes", 0),
    )

    probe_scale_candidates = sorted(
        {
            round(settings.robustness_budget_scale, 4),
            0.25,
            0.10,
        },
        reverse=True,
    )
    selected_train = settings.candidate_train_floor_timesteps
    selected_scale = min(probe_scale_candidates)
    for probe_scale in probe_scale_candidates:
        denominator = remaining_iterations * (1.0 + (settings.variant_count * probe_scale))
        proposed_train = _round_to_rollout_multiple(
            max(settings.candidate_train_floor_timesteps, remaining_train / denominator),
            settings.rollout_size,
        )
        if settings.candidate_train_ceiling_timesteps is not None:
            proposed_train = min(
                proposed_train,
                _round_to_rollout_multiple(
                    settings.candidate_train_ceiling_timesteps,
                    settings.rollout_size,
                ),
            )
        if _estimated_training_spend(
            proposed_train,
            probe_scale,
            settings.rollout_size,
            settings.variant_count,
        ) <= remaining_train:
            selected_train = proposed_train
            selected_scale = probe_scale
            break

    if remaining_iterations == 1:
        spend_all_train = remaining_train / (1.0 + (settings.variant_count * selected_scale))
        selected_train = max(
            settings.candidate_train_floor_timesteps,
            _round_to_rollout_multiple(spend_all_train, settings.rollout_size),
        )
        if settings.candidate_train_ceiling_timesteps is not None:
            selected_train = min(
                selected_train,
                _round_to_rollout_multiple(
                    settings.candidate_train_ceiling_timesteps,
                    settings.rollout_size,
                ),
            )

    remaining_eval_share = max(
        1.0,
        remaining_eval / (remaining_iterations * (1.0 + (settings.variant_count * selected_scale))),
    )
    reflection_checkpoints = min(
        settings.target_reflection_checkpoints,
        max(1, selected_train // settings.rollout_size),
    )
    reflection_episodes = min(
        settings.reflection_episode_ceiling,
        max(0, int(remaining_eval_share // max(reflection_checkpoints + 1, 2))),
    )
    final_eval_episodes = min(
        settings.evaluation_episode_ceiling,
        max(1, int(round(remaining_eval_share - (reflection_checkpoints * reflection_episodes)))),
    )
    while (
        (reflection_checkpoints * reflection_episodes) + final_eval_episodes
        > max(1, int(round(remaining_eval_share)))
        and reflection_episodes > 0
    ):
        reflection_episodes -= 1
    while (
        (reflection_checkpoints * reflection_episodes) + final_eval_episodes
        > max(1, int(round(remaining_eval_share)))
        and reflection_checkpoints > 1
    ):
        reflection_checkpoints -= 1
    reflection_interval_steps = _round_to_rollout_multiple(
        max(settings.rollout_size, selected_train / max(reflection_checkpoints, 1)),
        settings.rollout_size,
    )

    return AdaptiveBudgetPlan(
        primary_train_timesteps=selected_train,
        evaluation_episodes=final_eval_episodes,
        reflection_episodes=reflection_episodes,
        reflection_interval_steps=reflection_interval_steps,
        robustness_budget_scale=selected_scale,
        rationale=(
            "adaptive PPO plan from remaining session budget: "
            f"train={selected_train}, eval={final_eval_episodes}, "
            f"reflection={reflection_episodes}x{reflection_checkpoints}, "
            f"probe_scale={selected_scale:.2f}"
        ),
    )


def record_iteration_budget_usage(
    metadata: dict[str, Any],
    *,
    candidate_id: str,
    iteration_index: int,
    performance_metrics: dict[str, Any],
    robustness_runs: list[Any],
    llm_calls_used: int,
    plan: AdaptiveBudgetPlan | None,
) -> dict[str, Any]:
    """
    Update session metadata with actual spend from one completed iteration.
    """
    updated = dict(metadata)
    primary_train = _int_value(performance_metrics, "total_timesteps", 0, minimum=0)
    primary_eval = _int_value(
        performance_metrics,
        "evaluation_episodes_consumed",
        0,
        minimum=0,
    )
    probe_train = sum(
        _int_value(run.metrics, "total_timesteps", 0, minimum=0)
        for run in robustness_runs
    )
    probe_eval = sum(
        _int_value(run.metrics, "evaluation_episodes_consumed", 0, minimum=0)
        for run in robustness_runs
    )
    updated["spent_training_timesteps"] = (
        _int_value(updated, "spent_training_timesteps", 0, minimum=0)
        + primary_train
        + probe_train
    )
    updated["spent_evaluation_episodes"] = (
        _int_value(updated, "spent_evaluation_episodes", 0, minimum=0)
        + primary_eval
        + probe_eval
    )
    updated["llm_calls_used"] = (
        _int_value(updated, "llm_calls_used", 0, minimum=0)
        + max(0, llm_calls_used)
    )
    history = list(updated.get("budget_history", []))
    history.append(
        {
            "candidate_id": candidate_id,
            "iteration_index": iteration_index,
            "primary_train_timesteps": primary_train,
            "primary_evaluation_episodes": primary_eval,
            "probe_train_timesteps": probe_train,
            "probe_evaluation_episodes": probe_eval,
            "llm_calls_used": llm_calls_used,
            "plan": plan.as_metadata_overrides() if plan is not None else None,
        }
    )
    updated["budget_history"] = history
    return updated


def budget_exhausted_for_next_iteration(
    session: dict[str, Any],
    metadata: dict[str, Any],
    *,
    next_iteration_index: int,
) -> bool:
    """
    Report whether the remaining session budget can support another PPO iteration.
    """
    if EnvironmentBackend(session["environment_backend"]) is not EnvironmentBackend.GYMNASIUM:
        return False
    if _string_value(metadata, "execution_mode", "deterministic") != "ppo":
        return False
    settings = AdaptiveBudgetSettings.from_session(session, metadata)
    if not settings.is_adaptive:
        return False
    remaining_train = settings.total_training_timesteps - _int_value(
        metadata,
        "spent_training_timesteps",
        0,
        minimum=0,
    )
    remaining_eval = settings.total_evaluation_episodes - _int_value(
        metadata,
        "spent_evaluation_episodes",
        0,
        minimum=0,
    )
    return (
        remaining_train < settings.rollout_size
        or remaining_eval < 1
        or next_iteration_index >= session["max_iterations"]
    )


def remaining_budget_snapshot(
    session: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, int]:
    """
    Summarize remaining session budget after any iteration.
    """
    if EnvironmentBackend(session["environment_backend"]) is not EnvironmentBackend.GYMNASIUM:
        return {}
    if _string_value(metadata, "execution_mode", "deterministic") != "ppo":
        return {}
    settings = AdaptiveBudgetSettings.from_session(session, metadata)
    return {
        "remaining_training_timesteps": max(
            0,
            settings.total_training_timesteps
            - _int_value(metadata, "spent_training_timesteps", 0, minimum=0),
        ),
        "remaining_evaluation_episodes": max(
            0,
            settings.total_evaluation_episodes
            - _int_value(metadata, "spent_evaluation_episodes", 0, minimum=0),
        ),
        "remaining_llm_calls": max(
            0,
            settings.max_llm_calls - _int_value(metadata, "llm_calls_used", 0, minimum=0),
        ),
    }


def _variant_count(backend: EnvironmentBackend) -> int:
    """
    Resolve the configured robustness variant count for a backend.
    """
    runner = RobustnessRunner()
    return len(runner._variants_for_backend(backend.value))  # noqa: SLF001 - centralized helper.


def _estimated_training_spend(
    primary_train_timesteps: int,
    probe_scale: float,
    rollout_size: int,
    variant_count: int,
) -> int:
    """
    Estimate total PPO training spend for one primary run plus its probe variants.
    """
    probe_timesteps = _round_to_rollout_multiple(
        max(rollout_size, primary_train_timesteps * probe_scale),
        rollout_size,
    )
    return primary_train_timesteps + (variant_count * probe_timesteps)


def _round_to_rollout_multiple(value: float | int, rollout_size: int) -> int:
    """
    Round a positive value to the nearest PPO rollout multiple.
    """
    normalized = max(rollout_size, int(round(float(value))))
    return rollout_size * math.ceil(normalized / rollout_size)


def _string_value(mapping: dict[str, Any], key: str, default: str) -> str:
    """
    Normalize a string-like value from a metadata mapping.
    """
    value = mapping.get(key, default)
    text = str(value).strip().lower()
    return text or default


def _int_value(
    mapping: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """
    Normalize an integer value from a metadata mapping.
    """
    value = mapping.get(key, default)
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    if minimum is not None:
        return max(minimum, normalized)
    return normalized


def _optional_int_value(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    """
    Normalize an optional integer value from a metadata mapping.
    """
    if key not in mapping or mapping.get(key) in {None, ""}:
        return None
    return _int_value(mapping, key, int(mapping[key]), minimum=minimum)


def _float_value(
    mapping: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    """
    Normalize a float value from a metadata mapping.
    """
    value = mapping.get(key, default)
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = default
    if minimum is not None:
        return max(minimum, normalized)
    return normalized
