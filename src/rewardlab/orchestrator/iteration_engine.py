"""
Summary: Iteration engine implementing evaluate-reflect-revise candidate updates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.factory import resolve_backend_adapter
from rewardlab.schemas.session_config import EnvironmentBackend


@dataclass(slots=True, frozen=True)
class IterationResult:
    """
    Capture one complete iteration output for repository persistence.
    """

    reward_definition: str
    change_summary: str
    score: float
    performance_summary: str
    reflection_summary: str
    proposed_changes: list[str]
    confidence: float


class IterationEngine:
    """
    Execute deterministic local iteration logic for baseline session workflows.
    """

    def run_iteration(
        self,
        session: dict[str, Any],
        iteration_index: int,
        baseline_reward_definition: str,
    ) -> IterationResult:
        """
        Run one evaluate-reflect-revise iteration for a session.

        Args:
            session: Session metadata dictionary.
            iteration_index: Zero-based iteration index.
            baseline_reward_definition: Seed reward definition text.

        Returns:
            Iteration result payload.
        """
        environment_backend = EnvironmentBackend(session["environment_backend"])
        reward_definition = self._revise_reward_definition(
            baseline_reward_definition=baseline_reward_definition,
            iteration_index=iteration_index,
            environment_backend=environment_backend,
        )
        payload = ExperimentInput(
            session_id=session["session_id"],
            environment_id=session["environment_id"],
            environment_backend=environment_backend,
            reward_definition=reward_definition,
            iteration_index=iteration_index,
            objective_text=session["objective_text"],
        )
        adapter = resolve_backend_adapter(environment_backend)
        performance = adapter.run_performance(payload)
        reflection = adapter.run_reflection(payload)
        confidence = min(0.95, round(0.65 + (0.05 * iteration_index), 4))
        proposed_changes = [
            (
                f"Investigate {environment_backend.value} probe stability at "
                f"iteration {iteration_index + 1}."
            ),
        ]
        return IterationResult(
            reward_definition=reward_definition,
            change_summary=f"Iteration {iteration_index} reward revision",
            score=performance.score,
            performance_summary=performance.summary,
            reflection_summary=reflection.summary,
            proposed_changes=proposed_changes,
            confidence=confidence,
        )

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
