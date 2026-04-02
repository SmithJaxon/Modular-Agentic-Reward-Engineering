"""
Summary: Iteration engine implementing evaluate-reflect-revise candidate updates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rewardlab.experiments.backends.base import ExperimentInput, ExperimentOutput
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
        reward_definition = self._revise_reward_definition(
            baseline_reward_definition=baseline_reward_definition,
            iteration_index=iteration_index,
            environment_backend=EnvironmentBackend(session["environment_backend"]),
        )
        output = self._run_performance_probe(
            session_id=session["session_id"],
            environment_id=session["environment_id"],
            environment_backend=EnvironmentBackend(session["environment_backend"]),
            reward_definition=reward_definition,
            iteration_index=iteration_index,
            objective_text=session["objective_text"],
        )
        confidence = min(0.95, 0.65 + (0.05 * iteration_index))
        reflection_summary = (
            f"Iteration {iteration_index} scored {output.score:.3f}. "
            f"Propose reward shaping update focused on stability."
        )
        proposed_changes = [f"Adjust weighted penalties for iteration {iteration_index + 1}."]
        return IterationResult(
            reward_definition=reward_definition,
            change_summary=f"Iteration {iteration_index} reward revision",
            score=output.score,
            performance_summary=output.summary,
            reflection_summary=reflection_summary,
            proposed_changes=proposed_changes,
            confidence=confidence,
        )

    def _run_performance_probe(
        self,
        session_id: str,
        environment_id: str,
        environment_backend: EnvironmentBackend,
        reward_definition: str,
        iteration_index: int,
        objective_text: str,
    ) -> ExperimentOutput:
        """
        Produce a deterministic local performance estimate for testable iterations.
        """
        payload = ExperimentInput(
            session_id=session_id,
            environment_id=environment_id,
            environment_backend=environment_backend,
            reward_definition=reward_definition,
            iteration_index=iteration_index,
            objective_text=objective_text,
        )
        base = 0.55 + (0.08 * iteration_index)
        taper = max(0.0, (iteration_index - 4) * 0.03)
        score = round(min(0.97, base - taper), 4)
        summary = (
            f"{payload.environment_backend.value} run iteration {iteration_index}: "
            f"score={score:.3f}"
        )
        return ExperimentOutput(score=score, metrics={"score": score}, summary=summary)

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
