"""
Summary: Native MCP tool-invocation bridge for agentic tool actions.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from rewardlab.agentic.contracts import ControllerAction, ToolResult
from rewardlab.llm.openai_client import ChatMessage, OpenAIClient, ResponseRequest
from rewardlab.schemas.agent_experiment import (
    ActionType,
    AgentExperimentRecord,
    McpServerConfig,
)
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.utils.env import load_runtime_environment

ACTION_TOOL_NAME: dict[ActionType, str] = {
    ActionType.RUN_EXPERIMENT: "run_experiment",
    ActionType.PROPOSE_REWARD: "propose_reward_revision",
    ActionType.SUMMARIZE_RUN_ARTIFACTS: "summarize_run_artifacts",
    ActionType.VALIDATE_REWARD_PROGRAM: "validate_reward_program",
    ActionType.ESTIMATE_COST_AND_RISK: "estimate_cost_and_risk",
    ActionType.COMPARE_CANDIDATES: "compare_candidates",
    ActionType.REQUEST_HUMAN_FEEDBACK: "request_human_feedback",
    ActionType.STOP: "stop_or_continue_recommendation",
}


class McpToolInvoker:
    """Invoke one agentic action through a native MCP tool call."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store OpenAI client used for native MCP Responses API calls."""

        self.openai_client = openai_client or OpenAIClient()

    def execute_action(
        self,
        *,
        record: AgentExperimentRecord,
        action: ControllerAction,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
    ) -> ToolResult:
        """Execute one action through MCP and normalize output into `ToolResult`."""

        if not self.openai_client.has_credentials:
            return ToolResult(
                status="error",
                summary="MCP tool invocation requires OPENAI_API_KEY credentials.",
                payload={"reason": "missing_openai_credentials"},
            )

        servers = record.spec.tool_policy.mcp_servers
        if len(servers) == 0:
            return ToolResult(
                status="error",
                summary="No MCP servers are configured for this experiment spec.",
                payload={"reason": "missing_mcp_servers"},
            )

        requested_tool_name = ACTION_TOOL_NAME[action.action_type]
        selected_server = _select_server(servers=servers, tool_name=requested_tool_name)
        response = self.openai_client.response(
            ResponseRequest(
                model=record.spec.models.controller.model,
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "Execute exactly one MCP tool call and return no extra prose. "
                            "If a tool call succeeds, return a compact confirmation."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=_build_user_prompt(
                            record=record,
                            action=action,
                            candidates=candidates,
                            runs=runs,
                            tool_name=requested_tool_name,
                            server_label=selected_server.server_label,
                        ),
                    ),
                ),
                reasoning_effort=_coerce_reasoning(
                    record.spec.models.controller.reasoning_effort
                ),
                max_output_tokens=min(
                    record.spec.models.controller.max_completion_tokens,
                    record.spec.budgets.api.max_completion_tokens_per_call,
                ),
                tools=(_tool_param(selected_server, requested_tool_name),),
                tool_choice={"type": "mcp", "server_label": selected_server.server_label},
                parallel_tool_calls=False,
                max_tool_calls=1,
            )
        )
        mcp_call = _select_mcp_call(
            output_items=response.output_items,
            requested_tool_name=requested_tool_name,
            server_label=selected_server.server_label,
        )
        if mcp_call is None:
            return ToolResult(
                status="error",
                summary=(
                    "Responses API did not return an MCP call output for the requested tool."
                ),
                payload={
                    "requested_tool_name": requested_tool_name,
                    "server_label": selected_server.server_label,
                    "output_text": response.output_text,
                    "response_id": response.response_id,
                },
                consumed_tokens=response.total_tokens or 0,
            )

        error_value = mcp_call.get("error")
        if isinstance(error_value, str) and error_value.strip():
            return ToolResult(
                status="error",
                summary=f"MCP call failed: {error_value.strip()}",
                payload={
                    "requested_tool_name": requested_tool_name,
                    "server_label": selected_server.server_label,
                    "mcp_call": mcp_call,
                },
                consumed_tokens=response.total_tokens or 0,
            )

        normalized = _normalize_mcp_output(
            mcp_output=mcp_call.get("output"),
            requested_tool_name=requested_tool_name,
            server_label=selected_server.server_label,
            consumed_tokens=response.total_tokens or 0,
        )
        return normalized


def _select_server(
    *,
    servers: list[McpServerConfig],
    tool_name: str,
) -> McpServerConfig:
    """Select a configured server, preferring one that explicitly allows the tool."""

    for server in servers:
        if server.allowed_tools is None:
            continue
        if tool_name in server.allowed_tools:
            return server
    return servers[0]


def _tool_param(server: McpServerConfig, tool_name: str) -> dict[str, object]:
    """Build one MCP tool parameter payload for a Responses API request."""

    payload: dict[str, object] = {
        "type": "mcp",
        "server_label": server.server_label,
        "allowed_tools": [tool_name],
        "require_approval": server.require_approval,
    }
    if server.server_url is not None:
        payload["server_url"] = server.server_url
    if server.connector_id is not None:
        payload["connector_id"] = server.connector_id
    if server.server_description is not None:
        payload["server_description"] = server.server_description
    if len(server.headers) > 0:
        payload["headers"] = dict(server.headers)
    authorization = _authorization_for_server(server)
    if authorization is not None:
        payload["authorization"] = authorization
    return payload


def _authorization_for_server(server: McpServerConfig) -> str | None:
    """Resolve an optional MCP bearer token from configured environment variable."""

    env_name = server.authorization_env_var
    if env_name is None:
        return None
    env = load_runtime_environment()
    token = env.get(env_name)
    if token is None or not token.strip():
        return None
    return token.strip()


def _build_user_prompt(
    *,
    record: AgentExperimentRecord,
    action: ControllerAction,
    candidates: list[RewardCandidate],
    runs: list[ExperimentRun],
    tool_name: str,
    server_label: str,
) -> str:
    """Build a focused invocation prompt for one MCP tool execution."""

    latest_candidate_id = candidates[-1].candidate_id if len(candidates) > 0 else None
    latest_run_id = runs[-1].run_id if len(runs) > 0 else None
    return (
        f"Server label: {server_label}\n"
        f"Tool to call: {tool_name}\n"
        f"Experiment id: {record.experiment_id}\n"
        f"Objective: {record.spec.objective}\n"
        f"Latest candidate id: {latest_candidate_id}\n"
        f"Latest run id: {latest_run_id}\n"
        f"Controller rationale: {action.rationale}\n"
        f"Action input JSON: {json.dumps(action.action_input)}\n"
        "Invoke the MCP tool exactly once.\n"
        "The tool should produce JSON compatible with local ToolResult payloads.\n"
    )


def _select_mcp_call(
    *,
    output_items: list[dict[str, Any]],
    requested_tool_name: str,
    server_label: str,
) -> dict[str, Any] | None:
    """Select the best MCP call output item from response output."""

    mcp_items = [
        item
        for item in output_items
        if str(item.get("type")) == "mcp_call"
    ]
    if len(mcp_items) == 0:
        return None

    exact = next(
        (
            item
            for item in mcp_items
            if str(item.get("name")) == requested_tool_name
            and str(item.get("server_label")) == server_label
        ),
        None,
    )
    if exact is not None:
        return exact
    return mcp_items[0]


def _normalize_mcp_output(
    *,
    mcp_output: object,
    requested_tool_name: str,
    server_label: str,
    consumed_tokens: int,
) -> ToolResult:
    """Normalize MCP output text/object into one `ToolResult`."""

    parsed_output = _parse_mcp_output(mcp_output)
    if isinstance(parsed_output, dict):
        status_value = parsed_output.get("status")
        summary_value = parsed_output.get("summary")
        payload_value = parsed_output.get("payload")
        if (
            isinstance(status_value, str)
            and status_value.strip()
            and isinstance(summary_value, str)
            and summary_value.strip()
            and isinstance(payload_value, dict)
        ):
            return ToolResult(
                status=status_value.strip(),
                summary=summary_value.strip(),
                payload=payload_value,
                consumed_tokens=consumed_tokens,
                consumed_usd=_optional_nonnegative_float(parsed_output.get("consumed_usd")),
            )
        return ToolResult(
            status="ok",
            summary=f"MCP tool {requested_tool_name} executed on {server_label}.",
            payload={
                "requested_tool_name": requested_tool_name,
                "server_label": server_label,
                **parsed_output,
            },
            consumed_tokens=consumed_tokens,
        )
    return ToolResult(
        status="ok",
        summary=f"MCP tool {requested_tool_name} executed on {server_label}.",
        payload={
            "requested_tool_name": requested_tool_name,
            "server_label": server_label,
            "raw_output": parsed_output,
        },
        consumed_tokens=consumed_tokens,
    )


def _parse_mcp_output(value: object) -> object:
    """Parse MCP output payloads when they contain JSON-encoded text."""

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return ""
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return normalized
    return value


def _coerce_reasoning(value: str) -> Literal["minimal", "low", "medium", "high"]:
    """Return a valid reasoning effort value for Responses API requests."""

    normalized = value.strip().lower()
    if normalized in {"minimal", "low", "medium", "high"}:
        return cast(Literal["minimal", "low", "medium", "high"], normalized)
    return "medium"


def _optional_nonnegative_float(value: object) -> float:
    """Return a non-negative float from a loosely typed numeric field."""

    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    return 0.0
