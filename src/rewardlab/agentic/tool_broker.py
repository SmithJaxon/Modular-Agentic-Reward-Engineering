"""
Summary: Tool broker that dispatches controller actions to isolated worker executors.
Created: 2026-04-10
Last Updated: 2026-04-16
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.agentic.mcp_invoker import McpToolInvoker
from rewardlab.agentic.tools.compare_candidates import CompareCandidatesTool
from rewardlab.agentic.tools.estimate_cost_and_risk import EstimateCostAndRiskTool
from rewardlab.agentic.tools.propose_reward import ProposeRewardTool
from rewardlab.agentic.tools.request_human_feedback import RequestHumanFeedbackTool
from rewardlab.agentic.tools.run_experiment import RunExperimentTool
from rewardlab.agentic.tools.summarize_run_artifacts import SummarizeRunArtifactsTool
from rewardlab.agentic.tools.validate_reward_program import ValidateRewardProgramTool
from rewardlab.schemas.agent_experiment import (
    ActionType,
    AgentExperimentRecord,
    McpExecutionMode,
)
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate

ACTION_TOOL_NAME = {
    ActionType.RUN_EXPERIMENT: "run_experiment",
    ActionType.PROPOSE_REWARD: "propose_reward_revision",
    ActionType.SUMMARIZE_RUN_ARTIFACTS: "summarize_run_artifacts",
    ActionType.VALIDATE_REWARD_PROGRAM: "validate_reward_program",
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
        summarize_run_artifacts_tool: SummarizeRunArtifactsTool,
        validate_reward_program_tool: ValidateRewardProgramTool,
        estimate_cost_and_risk_tool: EstimateCostAndRiskTool,
        compare_candidates_tool: CompareCandidatesTool,
        request_human_feedback_tool: RequestHumanFeedbackTool,
        mcp_tool_invoker: McpToolInvoker | None = None,
    ) -> None:
        """Store tool executors used during autonomous control."""

        self.run_experiment_tool = run_experiment_tool
        self.propose_reward_tool = propose_reward_tool
        self.summarize_run_artifacts_tool = summarize_run_artifacts_tool
        self.validate_reward_program_tool = validate_reward_program_tool
        self.estimate_cost_and_risk_tool = estimate_cost_and_risk_tool
        self.compare_candidates_tool = compare_candidates_tool
        self.request_human_feedback_tool = request_human_feedback_tool
        self.mcp_tool_invoker = mcp_tool_invoker

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

        mcp_mode = record.spec.tool_policy.mcp_execution_mode
        if mcp_mode != McpExecutionMode.OFF and self.mcp_tool_invoker is not None:
            mcp_result = self.mcp_tool_invoker.execute_action(
                record=record,
                action=action,
                candidates=candidates,
                runs=runs,
            )
            if mcp_result.status == "ok":
                return mcp_result
            if mcp_mode == McpExecutionMode.REQUIRED:
                return mcp_result

        return self._execute_local_with_policy(
            record=record,
            action=action,
            candidates=candidates,
            runs=runs,
        )

    def _execute_local_with_policy(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
    ) -> ToolResult:
        """Execute one action locally with timeout and retry semantics."""

        retries = max(record.spec.tool_policy.max_retries_per_tool, 0)
        timeout_seconds = record.spec.tool_policy.default_timeout_seconds
        attempt_limit = retries + 1
        last_result: ToolResult | None = None

        for attempt_index in range(attempt_limit):
            result = self._execute_local_once_with_timeout(
                timeout_seconds=timeout_seconds,
                fn=lambda: self._execute_local_once(
                    action=action,
                    record=record,
                    candidates=candidates,
                    runs=runs,
                ),
            )
            if result.status == "ok":
                if attempt_index == 0:
                    return result
                payload = dict(result.payload)
                payload["retry_attempts_used"] = attempt_index
                return result.model_copy(update={"payload": payload})
            last_result = result

        assert last_result is not None
        payload = dict(last_result.payload)
        payload["retry_attempts_used"] = retries
        return last_result.model_copy(update={"payload": payload})

    def _execute_local_once_with_timeout(
        self,
        *,
        timeout_seconds: int,
        fn: Callable[[], ToolResult],
    ) -> ToolResult:
        """Execute one local tool call in an isolated worker with timeout."""

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rewardlab-tool") as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=max(timeout_seconds, 1))
            except FuturesTimeoutError:
                return ToolResult(
                    status="error",
                    summary=f"tool execution timed out after {timeout_seconds} seconds",
                    payload={"timeout_seconds": timeout_seconds},
                )
            except Exception as exc:
                return ToolResult(
                    status="error",
                    summary=f"tool execution raised: {exc}",
                    payload={"exception_type": type(exc).__name__},
                )

    def _execute_local_once(
        self,
        *,
        action: ControllerAction,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
    ) -> ToolResult:
        """Dispatch one action to the local in-process tool implementation."""

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
        if action.action_type == ActionType.SUMMARIZE_RUN_ARTIFACTS:
            return self.summarize_run_artifacts_tool.execute(
                record=record,
                candidates=candidates,
                runs=runs,
                action_input=action.action_input,
            )
        if action.action_type == ActionType.VALIDATE_REWARD_PROGRAM:
            return self.validate_reward_program_tool.execute(
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
