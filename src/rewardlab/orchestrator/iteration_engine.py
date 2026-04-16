"""
Summary: Deterministic evaluate-reflect-revise engine for offline-safe RewardLab sessions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rewardlab.orchestrator.reward_designer import (
    DeterministicRewardDesigner,
    RewardDesigner,
    RewardDesignRequest,
)
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.feedback_entry import FeedbackEntry
from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import EnvironmentBackend

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class IterationArtifacts:
    """Artifacts produced by a single deterministic iteration step."""

    candidate: RewardCandidate
    reflection: ReflectionRecord
    run_id: str


@dataclass(frozen=True)
class PlannedIteration:
    """Revised candidate plus run metadata prepared ahead of execution."""

    candidate: RewardCandidate
    run_id: str
    proposed_changes: list[str]


class IterationEngine:
    """Produce offline-safe reward revisions for the deterministic session loop."""

    def __init__(self, *, reward_designer: RewardDesigner | None = None) -> None:
        """Store the candidate-generation strategy used by the iteration loop."""

        self.reward_designer = reward_designer or DeterministicRewardDesigner()

    def evaluate_candidate(
        self,
        *,
        objective_text: str,
        reward_definition: str,
        iteration_index: int,
    ) -> float:
        """Score a reward candidate using deterministic text heuristics."""

        objective_terms = set(_tokenize(objective_text))
        reward_terms = set(_tokenize(reward_definition))
        overlap_bonus = len(objective_terms & reward_terms) * 0.15
        structure_bonus = min(reward_definition.count("\n") * 0.03, 0.3)
        iteration_bonus = min(iteration_index * 0.2, 1.2)
        return round(1.0 + overlap_bonus + structure_bonus + iteration_bonus, 4)

    def run_iteration(
        self,
        *,
        session_id: str,
        objective_text: str,
        environment_id: str,
        environment_backend: EnvironmentBackend,
        current_candidate: RewardCandidate,
        latest_reflection: ReflectionRecord | None = None,
        latest_run: ExperimentRun | None = None,
    ) -> IterationArtifacts:
        """Generate reflection and revised candidate artifacts for one iteration."""

        planned_iteration = self.plan_iteration(
            session_id=session_id,
            objective_text=objective_text,
            environment_id=environment_id,
            environment_backend=environment_backend,
            current_candidate=current_candidate,
            latest_reflection=latest_reflection,
            latest_run=latest_run,
        )
        next_iteration_index = planned_iteration.candidate.iteration_index
        reflection = ReflectionRecord(
            reflection_id=f"{session_id}-reflection-{next_iteration_index:03d}",
            candidate_id=current_candidate.candidate_id,
            source_run_ids=[planned_iteration.run_id],
            summary=(
                "The previous candidate can be improved by making the reward "
                "signal more explicit about stability and centered control."
            ),
            proposed_changes=planned_iteration.proposed_changes,
            confidence=min(0.55 + next_iteration_index * 0.08, 0.95),
        )
        revised_candidate = planned_iteration.candidate.model_copy(
            update={
                "aggregate_score": self.evaluate_candidate(
                    objective_text=objective_text,
                    reward_definition=planned_iteration.candidate.reward_definition,
                    iteration_index=next_iteration_index,
                )
            }
        )
        return IterationArtifacts(
            candidate=revised_candidate,
            reflection=reflection,
            run_id=planned_iteration.run_id,
        )

    def plan_iteration(
        self,
        *,
        session_id: str,
        objective_text: str,
        environment_id: str,
        environment_backend: EnvironmentBackend,
        current_candidate: RewardCandidate,
        latest_reflection: ReflectionRecord | None = None,
        latest_run: ExperimentRun | None = None,
        allowed_parameter_names: tuple[str, ...] = (),
    ) -> PlannedIteration:
        """Prepare the next revised candidate and its deterministic run identifier."""

        next_iteration_index = current_candidate.iteration_index + 1
        run_id = f"{session_id}-run-{next_iteration_index:03d}"
        design_result = self.reward_designer.design_next_candidate(
            RewardDesignRequest(
                session_id=session_id,
                objective_text=objective_text,
                environment_id=environment_id,
                environment_backend=environment_backend,
                current_candidate=current_candidate,
                next_iteration_index=next_iteration_index,
                latest_reflection=latest_reflection,
                latest_run=latest_run,
                allowed_parameter_names=allowed_parameter_names,
            )
        )
        revised_candidate = RewardCandidate(
            candidate_id=f"{session_id}-candidate-{next_iteration_index:03d}",
            session_id=session_id,
            parent_candidate_id=current_candidate.candidate_id,
            iteration_index=next_iteration_index,
            reward_definition=design_result.reward_definition,
            change_summary=design_result.change_summary,
            aggregate_score=None,
        )
        return PlannedIteration(
            candidate=revised_candidate,
            run_id=run_id,
            proposed_changes=design_result.proposed_changes,
        )

    def build_execution_reflection(
        self,
        *,
        session_id: str,
        candidate: RewardCandidate,
        run_id: str,
        metrics: dict[str, float | int | bool],
        proposed_changes: list[str],
    ) -> ReflectionRecord:
        """Build a reflection record that points at a completed backend execution run."""

        score = float(metrics.get("episode_reward", metrics.get("total_reward", 0.0)))
        step_count = int(metrics.get("step_count", 0))
        confidence = min(0.6 + candidate.iteration_index * 0.05, 0.95)
        return ReflectionRecord(
            reflection_id=f"{session_id}-reflection-{candidate.iteration_index:03d}",
            candidate_id=candidate.candidate_id,
            source_run_ids=[run_id],
            summary=(
                f"Run {run_id} produced score {score:.3f} over {step_count} steps. "
                "Use the backend evidence to keep improving stability and centered control."
            ),
            proposed_changes=proposed_changes,
            confidence=confidence,
        )


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized alphanumeric terms."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_iteration_summary(
    *,
    candidate: RewardCandidate,
    reflection: ReflectionRecord | None,
    feedback_entries: Sequence[FeedbackEntry],
) -> str:
    """Build a concise iteration summary that incorporates available feedback."""

    base_summary = reflection.summary if reflection is not None else candidate.change_summary
    if not feedback_entries:
        return base_summary
    return f"{base_summary} Feedback entries recorded: {len(feedback_entries)}."
