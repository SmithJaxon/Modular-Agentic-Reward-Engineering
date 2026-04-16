"""
Summary: Schemas for agentic decision-turn run configuration and decisions.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rewardlab.schemas.session_config import EnvironmentBackend


class ReasoningEffort(StrEnum):
    """
    Enumerate supported reasoning-effort guidance values.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentDecisionAction(StrEnum):
    """
    Enumerate top-level decision actions from the primary optimizer.
    """

    THINK = "think"
    REQUEST_TOOL = "request_tool"
    STOP = "stop"


class StopDecisionTag(StrEnum):
    """
    Enumerate normalized stop rationale tags.
    """

    OBJECTIVE_MET = "objective_met"
    PLATEAU = "plateau"
    COST_INEFFICIENT = "cost_inefficient"
    RISK_LIMIT = "risk_limit"
    MANUAL = "manual"
    TURN_CAP = "turn_cap"


class AgenticRunStatus(StrEnum):
    """
    Enumerate top-level lifecycle status for agentic runs.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EnvironmentSpec(BaseModel):
    """
    Define target environment settings for one agentic run.
    """

    model_config = ConfigDict(extra="forbid")

    backend: EnvironmentBackend
    id: str = Field(min_length=1)
    seed: int = 7

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        """
        Normalize and validate non-empty environment IDs.
        """
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("environment id cannot be blank")
        return trimmed


class ObjectiveSpec(BaseModel):
    """
    Define objective and baseline reward inputs.
    """

    model_config = ConfigDict(extra="forbid")

    text_file: str = Field(min_length=1)
    baseline_reward_file: str = Field(min_length=1)


class AgentProfileSpec(BaseModel):
    """
    Define model configuration for the primary optimizer agent.
    """

    model_config = ConfigDict(extra="forbid")

    primary_model: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    fallback_model: str = Field(min_length=1)
    planner_provider: str = Field(default="heuristic", min_length=1)
    planner_fallback_enabled: bool = True
    planner_system_prompt_file: str | None = None
    planner_context_window: int = Field(default=12, ge=1, le=200)
    planner_max_output_tokens: int = Field(default=1200, ge=128, le=4096)
    planner_max_retries: int = Field(default=1, ge=0, le=3)


class ToolPolicySpec(BaseModel):
    """
    Define allowed tool call policy for one run.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: tuple[str, ...] = Field(min_length=1)
    max_parallel_workers: int = Field(default=1, ge=1)
    per_call_timeout_seconds: int = Field(default=1200, ge=1)


class DecisionPolicySpec(BaseModel):
    """
    Define decision-turn limits for one run.
    """

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=10, ge=1)
    min_candidates_before_compare: int = Field(default=2, ge=2)
    compare_every_new_candidates: int = Field(default=2, ge=1)
    require_probe_before_compare: bool = True


class HardBudgetSpec(BaseModel):
    """
    Define hard budget limits for one run.
    """

    model_config = ConfigDict(extra="forbid")

    max_wall_clock_minutes: int = Field(ge=1)
    max_training_timesteps: int = Field(ge=1)
    max_evaluation_episodes: int = Field(ge=1)
    max_api_input_tokens: int = Field(ge=0)
    max_api_output_tokens: int = Field(ge=0)
    max_api_usd: float = Field(ge=0.0)
    max_calls_per_model: dict[str, int] = Field(default_factory=dict)


class SoftBudgetGuidanceSpec(BaseModel):
    """
    Define non-binding budget and stop guidance signals.
    """

    model_config = ConfigDict(extra="forbid")

    target_env_return: float | None = None
    plateau_window_turns: int = Field(default=3, ge=1)
    min_delta_return: float = 0.0
    min_gain_per_1k_usd: float = 0.0
    risk_ceiling: str = Field(default="high", min_length=1)


class BudgetSpec(BaseModel):
    """
    Define hard and soft budget settings for one run.
    """

    model_config = ConfigDict(extra="forbid")

    hard: HardBudgetSpec
    soft: SoftBudgetGuidanceSpec = Field(default_factory=SoftBudgetGuidanceSpec)


class TrainingDefaultsSpec(BaseModel):
    """
    Define default PPO knobs offered to experiment tools.
    """

    model_config = ConfigDict(extra="forbid")

    ppo_num_envs: int = Field(default=1, ge=1)
    ppo_total_timesteps: int = Field(default=4096, ge=64)
    ppo_n_steps: int = Field(default=128, ge=32)
    ppo_batch_size: int = Field(default=128, ge=32)
    ppo_learning_rate: float = Field(default=3e-4, gt=0.0)
    ppo_gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    ppo_gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    ppo_clip_range: float = Field(default=0.2, gt=0.0)
    ppo_n_epochs: int = Field(default=10, ge=1)
    ppo_ent_coef: float = Field(default=0.0, ge=0.0)
    ppo_vf_coef: float = Field(default=0.5, ge=0.0)
    ppo_max_grad_norm: float = Field(default=0.5, gt=0.0)
    ppo_activation_fn: str = Field(default="tanh", min_length=1)
    ppo_policy_hidden_sizes: tuple[int, ...] = Field(default=(64, 64), min_length=1)
    evaluation_episodes: int = Field(default=5, ge=1)
    reflection_episodes: int = Field(default=2, ge=0)
    reflection_interval_steps: int = Field(default=1024, ge=64)
    execution_mode: str = Field(default="deterministic", min_length=1)
    llm_provider: str = Field(default="none", min_length=1)
    llm_model: str = Field(default="gpt-4o-mini", min_length=1)


class ReportingSpec(BaseModel):
    """
    Define trace persistence preferences for one run.
    """

    model_config = ConfigDict(extra="forbid")

    save_decision_trace: bool = True
    save_tool_trace: bool = True
    save_budget_ledger: bool = True


class AgenticRunSpec(BaseModel):
    """
    Define the agentic run specification file contract.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    run_name: str = Field(min_length=1)
    environment: EnvironmentSpec
    objective: ObjectiveSpec
    agent: AgentProfileSpec
    tools: ToolPolicySpec
    decision: DecisionPolicySpec
    budgets: BudgetSpec
    training_defaults: TrainingDefaultsSpec = Field(default_factory=TrainingDefaultsSpec)
    reporting: ReportingSpec = Field(default_factory=ReportingSpec)


class AgentDecision(BaseModel):
    """
    Represent one primary-agent decision turn output.
    """

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    decision_source: str = Field(default="heuristic", min_length=1)
    action: AgentDecisionAction
    summary: str = Field(min_length=1)
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_rationale: str | None = None
    stop_tag: StopDecisionTag | None = None
    stop_reason: str | None = None

    @model_validator(mode="after")
    def _validate_action_payload(self) -> AgentDecision:
        """
        Enforce required fields for tool and stop decision actions.
        """
        if self.action is AgentDecisionAction.REQUEST_TOOL:
            if not self.tool_name:
                raise ValueError("tool_name is required for request_tool decisions")
            if not self.tool_rationale:
                raise ValueError("tool_rationale is required for request_tool decisions")
        if self.action is AgentDecisionAction.STOP:
            if self.stop_tag is None:
                raise ValueError("stop_tag is required for stop decisions")
            if not self.stop_reason:
                raise ValueError("stop_reason is required for stop decisions")
        return self
