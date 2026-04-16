"""
Summary: Stop-guidance heuristics for primary agent decisions.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

from dataclasses import dataclass

from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.cost_model import CostEfficiencyAssessment, assess_cost_efficiency
from rewardlab.schemas.agentic_run import AgenticRunSpec, StopDecisionTag
from rewardlab.schemas.budget_state import BudgetState


@dataclass(slots=True, frozen=True)
class StopGuidanceDecision:
    """
    Represent one normalized stop recommendation from guidance heuristics.
    """

    tag: StopDecisionTag
    reason: str
    summary: str


class StopGuidance:
    """
    Evaluate non-binding stop heuristics from scores, risk, and budget signals.
    """

    _RISK_LEVEL_ORDER: dict[str, int] = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    def evaluate(
        self,
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> StopGuidanceDecision | None:
        """
        Return a stop recommendation when objective, plateau, risk, or cost indicates stop.
        """
        objective = self._objective_decision(spec=spec, context=context)
        if objective is not None:
            return objective

        plateau = self._plateau_decision(spec=spec, context=context)
        if plateau is not None:
            return plateau

        risk = self._risk_decision(spec=spec, context=context)
        if risk is not None:
            return risk

        cost = self._cost_decision(spec=spec, context=context, budget_state=budget_state)
        if cost is not None:
            return cost

        return self._hard_budget_decision(budget_state=budget_state)

    @staticmethod
    def _objective_decision(
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
    ) -> StopGuidanceDecision | None:
        """
        Stop when configured target return is reached by any observed candidate.
        """
        target = spec.budgets.soft.target_env_return
        best_score = context.best_score
        if target is None or best_score is None:
            return None
        if best_score < target:
            return None
        return StopGuidanceDecision(
            tag=StopDecisionTag.OBJECTIVE_MET,
            reason=f"best score {best_score:.3f} reached target {target:.3f}",
            summary=f"Stopping because target env return {target:.3f} was reached.",
        )

    @staticmethod
    def _plateau_decision(
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
    ) -> StopGuidanceDecision | None:
        """
        Stop when recent score spread falls below configured minimum improvement.
        """
        window = spec.budgets.soft.plateau_window_turns
        scores = context.experiment_scores()
        if len(scores) < window:
            return None
        recent = scores[-window:]
        score_range = max(recent) - min(recent)
        if score_range >= spec.budgets.soft.min_delta_return:
            return None
        return StopGuidanceDecision(
            tag=StopDecisionTag.PLATEAU,
            reason=(
                "recent score range "
                f"{score_range:.4f} below min_delta_return "
                f"{spec.budgets.soft.min_delta_return:.4f}"
            ),
            summary="Stopping due to plateau in recent experiment scores.",
        )

    def _risk_decision(
        self,
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
    ) -> StopGuidanceDecision | None:
        """
        Stop when latest robustness risk exceeds configured risk ceiling.
        """
        ceiling = spec.budgets.soft.risk_ceiling.strip().lower()
        ceiling_rank = self._RISK_LEVEL_ORDER.get(ceiling)
        if ceiling_rank is None:
            return None

        latest_probe = context.latest_probe_result()
        if latest_probe is None:
            return None
        risk_raw = latest_probe.output.get("risk_level")
        if not isinstance(risk_raw, str):
            return None
        risk_level = risk_raw.strip().lower()
        risk_rank = self._RISK_LEVEL_ORDER.get(risk_level)
        if risk_rank is None or risk_rank <= ceiling_rank:
            return None
        return StopGuidanceDecision(
            tag=StopDecisionTag.RISK_LIMIT,
            reason=f"risk level {risk_level} exceeded risk ceiling {ceiling}",
            summary=(
                "Stopping due to robustness risk exceeding configured ceiling."
            ),
        )

    @staticmethod
    def _cost_decision(
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> StopGuidanceDecision | None:
        """
        Stop when score improvement per API cost falls below configured threshold.
        """
        threshold = spec.budgets.soft.min_gain_per_1k_usd
        if threshold <= 0.0:
            return None

        scores = context.experiment_scores()
        if len(scores) < 2:
            return None
        baseline_score = scores[0]
        score_gain = max(scores) - baseline_score
        assessment: CostEfficiencyAssessment = assess_cost_efficiency(
            score_gain=score_gain,
            api_cost_usd=budget_state.usage.api_cost_usd,
            minimum_gain_per_1k_usd=threshold,
        )
        if assessment.is_efficient or assessment.gain_per_1k_usd is None:
            return None
        return StopGuidanceDecision(
            tag=StopDecisionTag.COST_INEFFICIENT,
            reason=(
                f"gain_per_1k_usd {assessment.gain_per_1k_usd:.4f} below "
                f"minimum {assessment.minimum_gain_per_1k_usd:.4f}"
            ),
            summary=(
                "Stopping because observed score improvement per API spend is below guidance."
            ),
        )

    @staticmethod
    def _hard_budget_decision(
        *,
        budget_state: BudgetState,
    ) -> StopGuidanceDecision | None:
        """
        Stop early when hard budget dimensions are fully exhausted.
        """
        exhausted: list[str] = []
        if budget_state.remaining_wall_clock_minutes() <= 0.0:
            exhausted.append("wall_clock_minutes")
        if budget_state.max_api_usd > 0.0 and budget_state.remaining_api_usd() <= 0.0:
            exhausted.append("api_usd")
        if budget_state.remaining_training_timesteps() <= 0:
            exhausted.append("training_timesteps")
        if budget_state.remaining_evaluation_episodes() <= 0:
            exhausted.append("evaluation_episodes")
        if not exhausted:
            return None
        exhausted_text = ",".join(exhausted)
        return StopGuidanceDecision(
            tag=StopDecisionTag.COST_INEFFICIENT,
            reason=f"hard budget exhausted for: {exhausted_text}",
            summary="Stopping because hard budget limits are exhausted.",
        )
