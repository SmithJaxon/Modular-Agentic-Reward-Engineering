"""
Summary: Iteration engine implementing evaluate-reflect-revise candidate updates.
Created: 2026-04-02
Last Updated: 2026-04-06
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.factory import resolve_backend_adapter
from rewardlab.experiments.gymnasium_runtime import (
    GymnasiumExecutionConfig,
    RewardProgram,
    RewardProgramError,
    default_env_reward_program_source,
)
from rewardlab.llm.openai_client import OpenAIClient, OpenAIClientConfig
from rewardlab.llm.reward_prompting import (
    build_gymnasium_environment_context,
    build_initial_reward_prompt,
    build_revision_reward_prompt,
    extract_reward_program_source,
)
from rewardlab.schemas.session_config import EnvironmentBackend

_RUNTIME_METADATA_KEYS = {
    "artifact_root",
    "execution_mode",
    "llm_provider",
    "llm_model",
    "budget_mode",
    "max_llm_calls",
    "llm_calls_used",
    "ppo_total_timesteps",
    "ppo_num_envs",
    "ppo_n_steps",
    "ppo_batch_size",
    "ppo_learning_rate",
    "evaluation_episodes",
    "reflection_episodes",
    "reflection_interval_steps",
    "train_seed",
    "robustness_budget_scale",
    "planned_ppo_total_timesteps",
    "planned_evaluation_episodes",
    "planned_reflection_episodes",
    "planned_reflection_interval_steps",
    "planned_robustness_budget_scale",
    "planned_budget_rationale",
    "human_feedback_enabled",
    "peer_feedback_enabled",
}


@dataclass(slots=True, frozen=True)
class IterationResult:
    """
    Capture one complete iteration output for repository persistence.
    """

    reward_definition: str
    change_summary: str
    score: float
    performance_metrics: dict[str, Any]
    performance_summary: str
    reflection_summary: str
    proposed_changes: list[str]
    confidence: float
    feedback_summary: str = ""
    llm_calls_used: int = 0


class IterationEngine:
    """
    Execute deterministic or PPO-backed iteration logic for session workflows.
    """

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """
        Initialize optional LLM dependencies for reward synthesis.
        """
        self._openai_client = openai_client

    def run_iteration(
        self,
        session: dict[str, Any],
        iteration_index: int,
        baseline_reward_definition: str,
        feedback_summary: str = "",
        seed_reflection_summary: str = "",
    ) -> IterationResult:
        """
        Run one evaluate-reflect-revise iteration for a session.

        Args:
            session: Session metadata dictionary.
            iteration_index: Zero-based iteration index.
            baseline_reward_definition: Current seed reward definition text.
            feedback_summary: Optional latest feedback context to fold into revision.
            seed_reflection_summary: Optional summary of the current best candidate's behavior.

        Returns:
            Iteration result payload.
        """
        environment_backend = EnvironmentBackend(session["environment_backend"])
        overrides = self._runtime_overrides(session)
        feedback_text = feedback_summary.strip()

        if self._is_real_gymnasium_execution(environment_backend, overrides):
            reward_definition, change_summary, llm_calls_used = (
                self._prepare_real_reward_definition(
                    session=session,
                    iteration_index=iteration_index,
                    seed_reward_definition=baseline_reward_definition,
                    seed_reflection_summary=seed_reflection_summary,
                    feedback_summary=feedback_text,
                    overrides=overrides,
                )
            )
        else:
            llm_calls_used = 0
            reward_definition = self._revise_reward_definition(
                baseline_reward_definition=baseline_reward_definition,
                iteration_index=iteration_index,
                environment_backend=environment_backend,
            )
            change_summary = f"Iteration {iteration_index} reward revision"

        if feedback_text:
            change_summary = f"{change_summary} informed by feedback: {feedback_text}"

        payload = ExperimentInput(
            session_id=session["session_id"],
            environment_id=session["environment_id"],
            environment_backend=environment_backend,
            reward_definition=reward_definition,
            iteration_index=iteration_index,
            objective_text=session["objective_text"],
            overrides=overrides,
        )
        adapter = resolve_backend_adapter(environment_backend)
        performance = adapter.run_performance(payload)
        reflection = adapter.run_reflection(payload)
        proposed_changes = self._build_proposed_changes(
            environment_backend=environment_backend,
            reflection_summary=reflection.summary,
            feedback_summary=feedback_text,
            real_execution=self._is_real_gymnasium_execution(environment_backend, overrides),
        )
        return IterationResult(
            reward_definition=reward_definition,
            change_summary=change_summary,
            score=performance.score,
            performance_metrics=dict(performance.metrics),
            performance_summary=performance.summary,
            reflection_summary=reflection.summary,
            proposed_changes=proposed_changes,
            confidence=self._confidence_for_run(
                iteration_index=iteration_index,
                real_execution=self._is_real_gymnasium_execution(environment_backend, overrides),
                performance_metrics=performance.metrics,
            ),
            feedback_summary=feedback_text,
            llm_calls_used=llm_calls_used,
        )

    @staticmethod
    def _runtime_overrides(session: dict[str, Any]) -> dict[str, Any]:
        """
        Extract runtime settings from session metadata for backend execution.
        """
        metadata = dict(session.get("metadata", {}))
        return {key: metadata[key] for key in _RUNTIME_METADATA_KEYS if key in metadata}

    @staticmethod
    def _is_real_gymnasium_execution(
        environment_backend: EnvironmentBackend,
        overrides: dict[str, Any],
    ) -> bool:
        """
        Report whether the current session requests PPO-backed Gymnasium execution.
        """
        if environment_backend is not EnvironmentBackend.GYMNASIUM:
            return False
        config = GymnasiumExecutionConfig.from_overrides(overrides, variant_label="default")
        return config.is_real_execution

    @staticmethod
    def _llm_budget_remaining(session: dict[str, Any]) -> int | None:
        """
        Return the remaining session-level LLM-call budget when configured.
        """
        metadata = dict(session.get("metadata", {}))
        if "max_llm_calls" not in metadata:
            return None
        max_llm_calls = int(metadata.get("max_llm_calls", 0))
        llm_calls_used = int(metadata.get("llm_calls_used", 0))
        return max(0, max_llm_calls - llm_calls_used)

    def _prepare_real_reward_definition(
        self,
        *,
        session: dict[str, Any],
        iteration_index: int,
        seed_reward_definition: str,
        seed_reflection_summary: str,
        feedback_summary: str,
        overrides: dict[str, Any],
    ) -> tuple[str, str, int]:
        """
        Resolve the reward program to execute for a PPO-backed Gymnasium iteration.
        """
        config = GymnasiumExecutionConfig.from_overrides(overrides, variant_label="default")
        baseline_hint = str(
            dict(session.get("metadata", {})).get("baseline_reward_definition", "")
        ).strip()
        llm_budget_remaining = self._llm_budget_remaining(session)
        llm_allowed = llm_budget_remaining is None or llm_budget_remaining > 0

        if iteration_index == 0 and self._is_executable_reward(seed_reward_definition):
            return (
                self._normalize_executable_reward(seed_reward_definition),
                "Executed provided baseline reward program",
                0,
            )

        if iteration_index == 0 and (config.llm_provider != "openai" or not llm_allowed):
            if config.llm_provider == "openai" and not llm_allowed:
                return (
                    default_env_reward_program_source(),
                    "LLM budget exhausted before iteration 0; seeded PPO with environment reward",
                    0,
                )
            if baseline_hint:
                return (
                    default_env_reward_program_source(),
                    "Baseline reward was not executable; seeded PPO with environment reward",
                    0,
                )
            return (
                default_env_reward_program_source(),
                "No executable baseline was provided; seeded PPO with environment reward",
                0,
            )

        if config.llm_provider == "openai" and llm_allowed:
            if iteration_index == 0:
                prompt = build_initial_reward_prompt(
                    objective_text=session["objective_text"],
                    environment_id=session["environment_id"],
                    environment_context=build_gymnasium_environment_context(
                        session["environment_id"]
                    ),
                    baseline_hint=baseline_hint,
                )
                synthesized = self._synthesize_reward_program(
                    prompt=prompt,
                    llm_model=config.llm_model,
                    fallback_reward_definition=seed_reward_definition,
                )
                return synthesized, "LLM-generated initial reward program", 1

            prompt = build_revision_reward_prompt(
                objective_text=session["objective_text"],
                environment_id=session["environment_id"],
                environment_context=build_gymnasium_environment_context(
                    session["environment_id"]
                ),
                previous_reward_definition=seed_reward_definition,
                reflection_summary=seed_reflection_summary,
                feedback_summary=feedback_summary,
            )
            synthesized = self._synthesize_reward_program(
                prompt=prompt,
                llm_model=config.llm_model,
                fallback_reward_definition=seed_reward_definition,
            )
            return synthesized, "LLM-revised reward program using PPO reflection", 1

        if self._is_executable_reward(seed_reward_definition):
            reason = "Reused current best executable reward program without LLM revision"
            if config.llm_provider == "openai" and not llm_allowed:
                reason = "LLM budget exhausted; reused current best executable reward program"
            return (
                self._normalize_executable_reward(seed_reward_definition),
                reason,
                0,
            )
        fallback_reason = (
            "Current reward was not executable; reverted to environment reward baseline"
        )
        if config.llm_provider == "openai" and not llm_allowed:
            fallback_reason = (
                "LLM budget exhausted and current reward was not executable; "
                "reverted to environment reward baseline"
            )
        return (
            default_env_reward_program_source(),
            fallback_reason,
            0,
        )

    def _synthesize_reward_program(
        self,
        *,
        prompt: str,
        llm_model: str,
        fallback_reward_definition: str,
    ) -> str:
        """
        Request reward-program source from the configured LLM and validate it.
        """
        client = self._openai_client or OpenAIClient(OpenAIClientConfig(model=llm_model))
        response_text = client.generate_text(prompt, max_output_tokens=1200)
        try:
            reward_definition = extract_reward_program_source(response_text)
            RewardProgram(reward_definition)
            return reward_definition
        except RewardProgramError:
            if self._is_executable_reward(fallback_reward_definition):
                return self._normalize_executable_reward(fallback_reward_definition)
            raise

    @staticmethod
    def _is_executable_reward(reward_definition: str) -> bool:
        """
        Report whether reward text satisfies the Gymnasium reward-program contract.
        """
        try:
            RewardProgram(reward_definition)
        except RewardProgramError:
            return False
        return True

    @staticmethod
    def _normalize_executable_reward(reward_definition: str) -> str:
        """
        Normalize executable reward-program text without altering its semantics.
        """
        return RewardProgram(reward_definition).source

    @staticmethod
    def _build_proposed_changes(
        *,
        environment_backend: EnvironmentBackend,
        reflection_summary: str,
        feedback_summary: str,
        real_execution: bool,
    ) -> list[str]:
        """
        Derive the next recommended edits from reflection and feedback context.
        """
        if not real_execution:
            proposed_changes = [
                (
                    f"Investigate {environment_backend.value} probe stability before the next "
                    "iteration."
                ),
            ]
        else:
            proposed_changes = [
                "Rewrite reward components whose checkpoint values stay nearly constant.",
                "Rescale dominant reward terms when they overwhelm the other components.",
                "Favor edits that raise environment return without shortening episodes.",
            ]
            if "episode_length_mean=0.0000" in reflection_summary:
                proposed_changes.insert(
                    0,
                    "Rewrite the reward more aggressively because PPO is not completing episodes.",
                )
        normalized_feedback = feedback_summary.strip()
        if normalized_feedback:
            proposed_changes.insert(
                0,
                f"Address reviewer feedback before the next iteration: {normalized_feedback}",
            )
        return proposed_changes

    @staticmethod
    def _confidence_for_run(
        *,
        iteration_index: int,
        real_execution: bool,
        performance_metrics: dict[str, Any],
    ) -> float:
        """
        Estimate a confidence score for the stored reflection payload.
        """
        if not real_execution:
            return min(0.95, round(0.65 + (0.05 * iteration_index), 4))

        checkpoint_count = len(performance_metrics.get("reflection_checkpoints", []))
        runtime_available = bool(performance_metrics.get("runtime_available", False))
        base = 0.55 if runtime_available else 0.35
        confidence = base + min(0.20, checkpoint_count * 0.03) + min(0.10, iteration_index * 0.02)
        return min(0.96, round(confidence, 4))

    @staticmethod
    def _revise_reward_definition(
        baseline_reward_definition: str,
        iteration_index: int,
        environment_backend: EnvironmentBackend,
    ) -> str:
        """
        Build deterministic revision text for candidate reward definitions.
        """
        return (
            f"{baseline_reward_definition.strip()}\n"
            f"# revision={iteration_index}; backend={environment_backend.value}"
        )
