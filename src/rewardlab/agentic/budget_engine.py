"""
Summary: Multi-axis budget accounting for agentic decision-turn execution.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime

from rewardlab.schemas.agentic_run import AgenticRunSpec
from rewardlab.schemas.budget_state import BudgetState
from rewardlab.schemas.tool_contracts import ToolResult


class BudgetEngine:
    """
    Track budget usage and enforce hard budget limits for one run.
    """

    def __init__(
        self,
        state: BudgetState,
        *,
        start_time: datetime | None = None,
    ) -> None:
        """
        Initialize budget tracking state.
        """
        self._state = state
        self._start_time = start_time or datetime.now(UTC)

    @classmethod
    def from_spec(cls, spec: AgenticRunSpec) -> BudgetEngine:
        """
        Build budget state from an agentic run specification.
        """
        hard = spec.budgets.hard
        return cls(
            BudgetState(
                max_wall_clock_minutes=hard.max_wall_clock_minutes,
                max_training_timesteps=hard.max_training_timesteps,
                max_evaluation_episodes=hard.max_evaluation_episodes,
                max_api_input_tokens=hard.max_api_input_tokens,
                max_api_output_tokens=hard.max_api_output_tokens,
                max_api_usd=hard.max_api_usd,
                max_calls_per_model=hard.max_calls_per_model,
            )
        )

    @property
    def state(self) -> BudgetState:
        """
        Expose the mutable budget state model.
        """
        return self._state

    def refresh_wall_clock_usage(self) -> None:
        """
        Update consumed wall-clock minutes from the runtime start timestamp.
        """
        elapsed = datetime.now(UTC) - self._start_time
        self._state.usage.wall_clock_minutes = max(
            0.0,
            elapsed.total_seconds() / 60.0,
        )

    def remaining(self) -> dict[str, int | float | dict[str, int]]:
        """
        Return remaining budget capacities for all tracked dimensions.
        """
        self.refresh_wall_clock_usage()
        return self._state.remaining()

    def can_execute(
        self,
        *,
        tool_name: str | None = None,
        estimated_training_timesteps: int = 0,
        estimated_evaluation_episodes: int = 0,
    ) -> tuple[bool, str | None]:
        """
        Report whether another tool call is allowed under hard budget limits.
        """
        self.refresh_wall_clock_usage()
        if self._state.remaining_wall_clock_minutes() <= 0.0:
            return False, "wall-clock budget exhausted"
        if self._state.remaining_api_usd() <= 0.0 and self._state.max_api_usd > 0.0:
            return False, "api usd budget exhausted"
        compute_tools = {"run_experiment", "run_probe_suite"}
        if tool_name in compute_tools:
            if self._state.remaining_training_timesteps() <= 0:
                return False, "training timestep budget exhausted"
            if self._state.remaining_evaluation_episodes() <= 0:
                return False, "evaluation episode budget exhausted"
            if estimated_training_timesteps > self._state.remaining_training_timesteps():
                return (
                    False,
                    "requested tool exceeds remaining training timestep budget",
                )
            if estimated_evaluation_episodes > self._state.remaining_evaluation_episodes():
                return (
                    False,
                    "requested tool exceeds remaining evaluation episode budget",
                )
        return True, None

    def apply_tool_result(self, result: ToolResult) -> None:
        """
        Add one tool result's resource usage to budget state.
        """
        usage = self._state.usage
        usage.training_timesteps += result.training_timesteps
        usage.evaluation_episodes += result.evaluation_episodes
        usage.api_input_tokens += result.api_input_tokens
        usage.api_output_tokens += result.api_output_tokens
        usage.api_cost_usd += result.api_cost_usd
        if result.model_used:
            usage.calls_per_model[result.model_used] = (
                usage.calls_per_model.get(result.model_used, 0) + 1
            )
