"""
Summary: Schemas for multi-axis budget limits, usage, and remaining capacity.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BudgetUsage(BaseModel):
    """
    Track consumed budget dimensions for one run.
    """

    model_config = ConfigDict(extra="forbid")

    wall_clock_minutes: float = Field(default=0.0, ge=0.0)
    training_timesteps: int = Field(default=0, ge=0)
    evaluation_episodes: int = Field(default=0, ge=0)
    api_input_tokens: int = Field(default=0, ge=0)
    api_output_tokens: int = Field(default=0, ge=0)
    api_cost_usd: float = Field(default=0.0, ge=0.0)
    calls_per_model: dict[str, int] = Field(default_factory=dict)


class BudgetState(BaseModel):
    """
    Capture hard limits and observed usage for one run.
    """

    model_config = ConfigDict(extra="forbid")

    max_wall_clock_minutes: int = Field(ge=1)
    max_training_timesteps: int = Field(ge=1)
    max_evaluation_episodes: int = Field(ge=1)
    max_api_input_tokens: int = Field(ge=0)
    max_api_output_tokens: int = Field(ge=0)
    max_api_usd: float = Field(ge=0.0)
    max_calls_per_model: dict[str, int] = Field(default_factory=dict)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)

    def remaining_wall_clock_minutes(self) -> float:
        """
        Return remaining wall-clock budget in minutes.
        """
        return max(0.0, float(self.max_wall_clock_minutes) - self.usage.wall_clock_minutes)

    def remaining_training_timesteps(self) -> int:
        """
        Return remaining training-timestep budget.
        """
        return max(0, self.max_training_timesteps - self.usage.training_timesteps)

    def remaining_evaluation_episodes(self) -> int:
        """
        Return remaining evaluation-episode budget.
        """
        return max(0, self.max_evaluation_episodes - self.usage.evaluation_episodes)

    def remaining_api_input_tokens(self) -> int:
        """
        Return remaining API input token budget.
        """
        return max(0, self.max_api_input_tokens - self.usage.api_input_tokens)

    def remaining_api_output_tokens(self) -> int:
        """
        Return remaining API output token budget.
        """
        return max(0, self.max_api_output_tokens - self.usage.api_output_tokens)

    def remaining_api_usd(self) -> float:
        """
        Return remaining API USD budget.
        """
        return max(0.0, self.max_api_usd - self.usage.api_cost_usd)

    def remaining_calls_per_model(self) -> dict[str, int]:
        """
        Return remaining model call quotas by model name.
        """
        remaining_calls: dict[str, int] = {}
        for model_name, limit in self.max_calls_per_model.items():
            used = self.usage.calls_per_model.get(model_name, 0)
            remaining_calls[model_name] = max(0, limit - used)
        return remaining_calls

    def remaining(self) -> dict[str, int | float | dict[str, int]]:
        """
        Return remaining budget values for all tracked dimensions.
        """
        return {
            "remaining_wall_clock_minutes": self.remaining_wall_clock_minutes(),
            "remaining_training_timesteps": self.remaining_training_timesteps(),
            "remaining_evaluation_episodes": self.remaining_evaluation_episodes(),
            "remaining_api_input_tokens": self.remaining_api_input_tokens(),
            "remaining_api_output_tokens": self.remaining_api_output_tokens(),
            "remaining_api_usd": self.remaining_api_usd(),
            "remaining_calls_per_model": self.remaining_calls_per_model(),
        }
