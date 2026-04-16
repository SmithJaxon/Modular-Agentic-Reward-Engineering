"""
Summary: Tool broker that dispatches controller actions to isolated worker executors.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.tools.compare_candidates import CompareCandidatesTool
from rewardlab.agentic.tools.estimate_cost_and_risk import EstimateCostAndRiskTool
from rewardlab.agentic.tools.propose_reward import ProposeRewardTool
from rewardlab.agentic.tools.request_human_feedback import RequestHumanFeedbackTool
from rewardlab.agentic.tools.run_experiment import RunExperimentTool
from rewardlab.schemas.agent_experiment import ActionType, AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate

ACTION_TOOL_NAME = {
    ActionType.RUN_EXPERIMENT: "run_experiment",
    ActionType.PROPOSE_REWARD: "propose_reward_revision",
    ActionType.ESTIMATE_COST_AND_RISK: "estimate_cost_and_risk",
    ActionType.COMPARE_CANDIDATES: "compare_candidates",
    ActionType.REQUEST_HUMAN_FEEDBACK: "request_human_feedback",
    ActionType.STOP: "stop_or_continue_recommendation",
}


class ToolBroker:
    """Dispatch controller actions to tool executors under policy constraints."""

    def __init__(
        self,
        *,
        run_experiment_tool: RunExperimentTool,
        propose_reward_tool: ProposeRewardTool,
        estimate_cost_and_risk_tool: EstimateCostAndRiskTool,
        compare_candidates_tool: CompareCandidatesTool,
        request_human_feedback_tool: RequestHumanFeedbackTool,
    ) -> None:
        """Store tool executors used during autonomous control."""

        self.run_experiment_tool = run_experiment_tool
        self.propose_reward_tool = propose_reward_tool
        self.estimate_cost_and_risk_tool = estimate_cost_and_risk_tool
        self.compare_candidates_tool = compare_candidates_tool
        self.request_human_feedback_tool = request_human_feedback_tool

    def execute_action(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
    ) -> ToolResult:
        """Execute one controller action and return a normalized tool result."""

        tool_name = ACTION_TOOL_NAME[action.action_type]
        if tool_name not in record.spec.tool_policy.allowed_tools:
            return ToolResult(
                status="error",
                summary=f"tool {tool_name!r} is not allowed by experiment policy",
                payload={"tool_name": tool_name},
            )

        if action.action_type == ActionType.STOP:
            return ToolResult(
                status="ok",
                summary="Controller selected stop action.",
                payload={"stop": True},
            )
        if action.action_type == ActionType.RUN_EXPERIMENT:
            return self.run_experiment_tool.execute(
                record=record,
                candidates=candidates,
                action_input=action.action_input,
                run_count=len(runs),
            )
        if action.action_type == ActionType.PROPOSE_REWARD:
            return self.propose_reward_tool.execute(
                record=record,
                candidates=candidates,
                runs=runs,
                action_input=action.action_input,
            )
        if action.action_type == ActionType.ESTIMATE_COST_AND_RISK:
            return self.estimate_cost_and_risk_tool.execute(
                record=record,
                candidates=candidates,
                runs=runs,
                action_input=action.action_input,
            )
        if action.action_type == ActionType.COMPARE_CANDIDATES:
            return self.compare_candidates_tool.execute(
                record=record,
                candidates=candidates,
                runs=runs,
                action_input=action.action_input,
            )
        if action.action_type == ActionType.REQUEST_HUMAN_FEEDBACK:
            return self.request_human_feedback_tool.execute(
                record=record,
                candidates=candidates,
                runs=runs,
                action_input=action.action_input,
            )
        return ToolResult(
            status="error",
            summary=f"unsupported action_type: {action.action_type.value}",
            payload={"action_type": action.action_type.value},
        )
