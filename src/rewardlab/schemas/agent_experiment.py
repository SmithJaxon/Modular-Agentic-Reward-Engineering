"""
Summary: Schemas for agent tool-calling experiment definitions and runtime state.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate

MetadataValue = str | int | float | bool

__all__ = [
    "ActionType",
    "AgentBudgetLedger",
    "AgentDecisionRecord",
    "AgentExperimentRecord",
    "AgentExperimentSpec",
    "AgentExperimentStatus",
    "ApiBudgetConfig",
    "BaselineRewardConfig",
    "ComputeBudgetConfig",
    "EnvironmentConfig",
    "ExecutionConfig",
    "ExecutionPpoConfig",
    "ExecutionRolloutConfig",
    "GovernanceConfig",
    "HumanFeedbackPolicy",
    "ModelConfig",
    "ModelSetConfig",
    "OutputConfig",
    "StoppingPolicyConfig",
    "TimeBudgetConfig",
    "ToolPolicyConfig",
]


class AgentExperimentStatus(StrEnum):
    """Lifecycle states for an autonomous tool-calling experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ActionType(StrEnum):
    """Actions the controller may request in one decision step."""

    RUN_EXPERIMENT = "run_experiment"
    RUN_ROBUSTNESS_PROBES = "run_robustness_probes"
    PROPOSE_REWARD = "propose_reward"
    SUMMARIZE_RUN_ARTIFACTS = "summarize_run_artifacts"
    VALIDATE_REWARD_PROGRAM = "validate_reward_program"
    ESTIMATE_COST_AND_RISK = "estimate_cost_and_risk"
    COMPARE_CANDIDATES = "compare_candidates"
    REQUEST_HUMAN_FEEDBACK = "request_human_feedback"
    STOP = "stop"


class EnvironmentConfig(BaseModel):
    """Environment selection for an autonomous experiment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    backend: EnvironmentBackend = EnvironmentBackend.GYMNASIUM
    id: str = Field(min_length=1)
    seed: int | None = None


class BaselineRewardConfig(BaseModel):
    """Baseline reward configuration for the first candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: str = "file"
    path: str = Field(min_length=1)
    entrypoint_name: str = "reward"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """Restrict baseline mode to the currently supported input type."""

        normalized = value.strip().lower()
        if normalized != "file":
            raise ValueError("baseline_reward.mode must be 'file'")
        return normalized


class ModelConfig(BaseModel):
    """Model/runtime parameters for one model role."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(min_length=1)
    reasoning_effort: str = "medium"
    max_completion_tokens: int = Field(default=4_000, ge=1)

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, value: str) -> str:
        """Restrict reasoning effort to supported values."""

        normalized = value.strip().lower()
        if normalized not in {"minimal", "low", "medium", "high"}:
            raise ValueError(
                "reasoning_effort must be one of 'minimal', 'low', 'medium', or 'high'"
            )
        return normalized


class ModelSetConfig(BaseModel):
    """Model choices for controller, reward-design, and analysis roles."""

    model_config = ConfigDict(extra="forbid")

    controller: ModelConfig
    reward_designer: ModelConfig
    analyzer: ModelConfig


class ApiBudgetConfig(BaseModel):
    """API budget constraints."""

    model_config = ConfigDict(extra="forbid")

    max_total_tokens: int = Field(default=250_000, ge=1)
    max_total_usd: float = Field(default=10.0, ge=0.0)
    max_completion_tokens_per_call: int = Field(default=8_000, ge=1)


class TimeBudgetConfig(BaseModel):
    """Wall-clock budget constraints."""

    model_config = ConfigDict(extra="forbid")

    max_wall_clock_minutes: int = Field(default=240, ge=1)


class ComputeBudgetConfig(BaseModel):
    """Compute budget constraints."""

    model_config = ConfigDict(extra="forbid")

    max_experiments: int = Field(default=10, ge=1)
    max_total_train_timesteps: int = Field(default=500_000, ge=0)
    max_parallel_experiments: int = Field(default=1, ge=1)


class BudgetConfig(BaseModel):
    """Top-level budget constraints."""

    model_config = ConfigDict(extra="forbid")

    api: ApiBudgetConfig
    time: TimeBudgetConfig
    compute: ComputeBudgetConfig


class StoppingPolicyConfig(BaseModel):
    """Controller stop-policy parameters."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=8, ge=1)
    plateau_window: int = Field(default=3, ge=1)
    min_relative_improvement: float = Field(default=0.02, ge=0.0)
    max_no_improve_streak: int = Field(default=3, ge=1)
    max_failed_actions: int = Field(default=3, ge=1)


class HumanFeedbackPolicy(BaseModel):
    """Policy controlling optional human feedback requests."""

    model_config = ConfigDict(extra="forbid")

    allow: bool = False
    feedback_gate: FeedbackGate = FeedbackGate.NONE
    max_requests: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_allowance(self) -> HumanFeedbackPolicy:
        """Keep feedback gate semantics coherent with allow flag."""

        if not self.allow and self.max_requests != 0:
            raise ValueError("human_feedback.max_requests must be 0 when allow is false")
        if not self.allow and self.feedback_gate != FeedbackGate.NONE:
            raise ValueError("human_feedback.feedback_gate must be 'none' when allow is false")
        return self


class GovernanceConfig(BaseModel):
    """Policy controls for stopping behavior and optional feedback."""

    model_config = ConfigDict(extra="forbid")

    stopping: StoppingPolicyConfig
    human_feedback: HumanFeedbackPolicy


class ToolPolicyConfig(BaseModel):
    """Tool-allowlist and execution policy for worker actions."""

    model_config = ConfigDict(extra="forbid")

    allowed_tools: list[str] = Field(min_length=1)
    default_timeout_seconds: int = Field(default=1_800, ge=1)
    max_retries_per_tool: int = Field(default=2, ge=0)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, value: list[str]) -> list[str]:
        """Normalize, deduplicate, and reject blank tool names."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = item.strip()
            if not cleaned:
                raise ValueError("allowed_tools must not include blank names")
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        required_tools = {
            "run_experiment",
            "propose_reward_revision",
            "summarize_run_artifacts",
            "validate_reward_program",
            "estimate_cost_and_risk",
            "compare_candidates",
            "stop_or_continue_recommendation",
        }
        missing = sorted(required_tools - set(normalized))
        if missing:
            joined = ", ".join(repr(item) for item in missing)
            raise ValueError(f"allowed_tools is missing required entries: {joined}")
        return normalized


class ExecutionPpoConfig(BaseModel):
    """PPO execution controls for Humanoid-style experiments."""

    model_config = ConfigDict(extra="forbid")

    total_timesteps: int = Field(default=50_000, ge=1)
    eval_runs: int = Field(default=5, ge=1)
    checkpoint_count: int = Field(default=10, ge=1)
    eval_episodes_per_checkpoint: int = Field(default=1, ge=1)


class ExecutionRolloutConfig(BaseModel):
    """Rollout execution controls for lightweight environments."""

    model_config = ConfigDict(extra="forbid")

    max_episode_steps: int = Field(default=200, ge=1)


class ExecutionConfig(BaseModel):
    """Execution configuration by environment family."""

    model_config = ConfigDict(extra="forbid")

    ppo: ExecutionPpoConfig | None = None
    rollout: ExecutionRolloutConfig | None = None


class OutputConfig(BaseModel):
    """Output/reporting options for the autonomous run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    runtime_dir: str = ".rewardlab"
    report_detail: str = "full"
    save_decision_trace: bool = True

    @field_validator("report_detail")
    @classmethod
    def validate_report_detail(cls, value: str) -> str:
        """Restrict report detail mode to known levels."""

        normalized = value.strip().lower()
        if normalized not in {"summary", "full"}:
            raise ValueError("report_detail must be either 'summary' or 'full'")
        return normalized


class AgentExperimentSpec(BaseModel):
    """Top-level autonomous experiment spec."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    version: int = Field(default=1, ge=1)
    experiment_name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    environment: EnvironmentConfig
    baseline_reward: BaselineRewardConfig
    models: ModelSetConfig
    budgets: BudgetConfig
    governance: GovernanceConfig
    tool_policy: ToolPolicyConfig
    execution: ExecutionConfig
    outputs: OutputConfig

    @field_validator("experiment_name", "objective")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Reject required text that becomes blank after trimming."""

        if not value:
            raise ValueError("value must not be blank")
        return value


class AgentBudgetLedger(BaseModel):
    """Runtime budget ledger updated on each decision/action."""

    model_config = ConfigDict(extra="forbid")

    consumed_total_tokens: int = 0
    consumed_total_usd: float = 0.0
    consumed_experiments: int = 0
    consumed_train_timesteps: int = 0
    consumed_human_feedback_requests: int = 0


class AgentExperimentRecord(BaseModel):
    """Persisted autonomous experiment state."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    status: AgentExperimentStatus
    spec: AgentExperimentSpec
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    stop_reason: str | None = None
    best_candidate_id: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    budget_ledger: AgentBudgetLedger = Field(default_factory=AgentBudgetLedger)

    @field_validator("experiment_id")
    @classmethod
    def reject_blank_experiment_id(cls, value: str) -> str:
        """Reject blank experiment identifiers."""

        if not value:
            raise ValueError("experiment_id must not be blank")
        return value


class AgentDecisionRecord(BaseModel):
    """Persisted action decision and tool result trace entry."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    action_type: ActionType
    rationale: str = Field(min_length=1)
    expected_value: float | None = None
    expected_cost: float | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    result_status: str = Field(min_length=1)
    result_summary: str = Field(min_length=1)
    result_payload: dict[str, Any] = Field(default_factory=dict)
    consumed_tokens: int = 0
    consumed_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("decision_id", "experiment_id", "rationale", "result_status", "result_summary")
    @classmethod
    def reject_blank_required(cls, value: str) -> str:
        """Reject blank required text fields."""

        if not value:
            raise ValueError("value must not be blank")
        return value
