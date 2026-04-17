"""
Summary: Budget and stopping policy evaluation for autonomous agent experiments.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.reward_candidate import RewardCandidate


@dataclass(frozen=True)
class PolicyDecision:
    """Represents a policy-evaluation result for stop/continue checks."""

    should_stop: bool
    reason: str


class PolicyEngine:
    """Evaluate hard budget limits and stop heuristics for an experiment loop."""

    def evaluate_stop(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        failed_actions: int,
        non_progress_actions: int = 0,
    ) -> PolicyDecision:
        """Return whether the loop must stop according to policy constraints."""

        spec = record.spec
        ledger = record.budget_ledger
        stopping = spec.governance.stopping
        now = datetime.now(UTC)

        if ledger.consumed_total_tokens >= spec.budgets.api.max_total_tokens:
            return PolicyDecision(True, "api_token_budget_exhausted")
        if ledger.consumed_total_usd >= spec.budgets.api.max_total_usd:
            return PolicyDecision(True, "api_cost_budget_exhausted")
        if ledger.consumed_experiments >= spec.budgets.compute.max_experiments:
            return PolicyDecision(True, "compute_experiment_budget_exhausted")
        if (
            spec.budgets.compute.max_total_train_timesteps > 0
            and ledger.consumed_train_timesteps >= spec.budgets.compute.max_total_train_timesteps
        ):
            return PolicyDecision(True, "compute_timesteps_budget_exhausted")
        if ledger.consumed_reward_generations >= spec.budgets.compute.max_reward_generations:
            return PolicyDecision(True, "reward_generation_budget_exhausted")
        if failed_actions >= stopping.max_failed_actions:
            return PolicyDecision(True, "failed_action_threshold_reached")
        if non_progress_actions >= max(stopping.plateau_window, 2) * 2:
            return PolicyDecision(True, "non_progress_action_threshold_reached")

        started_at = record.started_at or record.created_at
        elapsed_minutes = max((now - started_at).total_seconds() / 60.0, 0.0)
        if elapsed_minutes >= spec.budgets.time.max_wall_clock_minutes:
            return PolicyDecision(True, "wall_clock_budget_exhausted")

        latest_iteration = max((candidate.iteration_index for candidate in candidates), default=0)
        if latest_iteration >= stopping.max_iterations:
            return PolicyDecision(True, "iteration_cap_reached")

        evaluated = [candidate for candidate in candidates if candidate.aggregate_score is not None]
        evaluated.sort(key=lambda candidate: candidate.iteration_index)
        if len(evaluated) < 2:
            return PolicyDecision(False, "continue")

        no_improve_streak = _no_improve_streak(evaluated)
        if no_improve_streak >= stopping.max_no_improve_streak:
            return PolicyDecision(True, "no_improve_streak_reached")

        plateau_window = min(stopping.plateau_window, len(evaluated))
        window = evaluated[-plateau_window:]
        relative_improvement = _relative_improvement(window)
        if plateau_window >= 2 and relative_improvement < stopping.min_relative_improvement:
            return PolicyDecision(True, "plateau_detected")

        return PolicyDecision(False, "continue")


def _no_improve_streak(candidates: list[RewardCandidate]) -> int:
    """Return trailing no-improvement streak length for evaluated candidates."""

    best_score = float("-inf")
    streak = 0
    for candidate in candidates:
        assert candidate.aggregate_score is not None
        score = candidate.aggregate_score
        if score > best_score:
            best_score = score
            streak = 0
        else:
            streak += 1
    return streak


def _relative_improvement(window: list[RewardCandidate]) -> float:
    """Compute relative improvement between first and best score in a score window."""

    if not window:
        return 0.0
    first_score = window[0].aggregate_score or 0.0
    best_score = max(candidate.aggregate_score or 0.0 for candidate in window)
    denominator = max(abs(first_score), 1e-6)
    return max(best_score - first_score, 0.0) / denominator
