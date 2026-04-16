"""
Summary: Permissive tool broker for request validation, routing, and accounting.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from rewardlab.agentic.budget_engine import BudgetEngine
from rewardlab.schemas.tool_contracts import ToolRequest, ToolResult, ToolResultStatus
from rewardlab.tools.executors import (
    budget_snapshot_executor,
    compare_candidates_executor,
    export_report_executor,
    read_artifact_executor,
    run_experiment_executor,
    run_probe_suite_executor,
)
from rewardlab.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    """
    Build the default registry with scaffold executors.
    """
    registry = ToolRegistry()
    registry.register(
        name="budget_snapshot",
        version="1.0",
        description="Return current remaining budget dimensions.",
        executor=budget_snapshot_executor,
    )
    registry.register(
        name="read_artifact",
        version="1.0",
        description="Read a bounded text preview from a local artifact path.",
        executor=read_artifact_executor,
    )
    registry.register(
        name="run_experiment",
        version="0.2",
        description="Run one experiment candidate through the backend adapter interface.",
        executor=run_experiment_executor,
    )
    registry.register(
        name="run_probe_suite",
        version="0.2",
        description="Run robustness probes and return risk analysis for a candidate.",
        executor=run_probe_suite_executor,
    )
    registry.register(
        name="compare_candidates",
        version="0.2",
        description="Rank candidate snapshots and select the highest aggregate candidate.",
        executor=compare_candidates_executor,
    )
    registry.register(
        name="export_report",
        version="0.2",
        description="Export a provided report payload to a local JSON artifact.",
        executor=export_report_executor,
    )
    return registry


class ToolBroker:
    """
    Validate and route tool requests while tracking budget consumption.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        budget_engine: BudgetEngine,
        enabled_tools: tuple[str, ...],
    ) -> None:
        """
        Initialize broker dependencies for one run.
        """
        self._registry = registry
        self._budget_engine = budget_engine
        self._enabled_tools = set(enabled_tools)

    def execute(
        self,
        *,
        turn_index: int,
        tool_name: str,
        arguments: dict[str, object],
        rationale: str,
    ) -> ToolResult:
        """
        Execute one brokered tool request and return a normalized result.
        """
        request = ToolRequest(
            request_id=f"toolreq-{uuid4().hex[:12]}",
            turn_index=turn_index,
            tool_name=tool_name,
            arguments=arguments,
            rationale=rationale,
            requested_at=datetime.now(UTC).isoformat(),
        )
        if tool_name not in self._enabled_tools:
            return ToolResult(
                request_id=request.request_id,
                turn_index=turn_index,
                tool_name=tool_name,
                status=ToolResultStatus.REJECTED,
                error=f"tool {tool_name} is not in enabled allowlist",
            )

        estimated_training_timesteps, estimated_evaluation_episodes = (
            _estimate_budget_impact(tool_name=tool_name, arguments=arguments)
        )
        allowed, reason = self._budget_engine.can_execute(
            tool_name=tool_name,
            estimated_training_timesteps=estimated_training_timesteps,
            estimated_evaluation_episodes=estimated_evaluation_episodes,
        )
        if not allowed:
            return ToolResult(
                request_id=request.request_id,
                turn_index=turn_index,
                tool_name=tool_name,
                status=ToolResultStatus.REJECTED,
                error=reason or "budget exhausted",
            )

        manifest = self._registry.get(tool_name)
        if manifest is None:
            return ToolResult(
                request_id=request.request_id,
                turn_index=turn_index,
                tool_name=tool_name,
                status=ToolResultStatus.REJECTED,
                error=f"tool {tool_name} is not registered",
            )

        try:
            result = manifest.executor(request, self._budget_engine.state)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                request_id=request.request_id,
                turn_index=turn_index,
                tool_name=tool_name,
                status=ToolResultStatus.FAILED,
                error=f"tool execution failed ({exc.__class__.__name__}): {exc}",
            )
        self._budget_engine.apply_tool_result(result)
        return result


def _estimate_budget_impact(*, tool_name: str, arguments: dict[str, object]) -> tuple[int, int]:
    """
    Estimate per-call training/evaluation usage for pre-dispatch budget checks.
    """
    overrides_raw = arguments.get("overrides", {})
    overrides = overrides_raw if isinstance(overrides_raw, dict) else {}
    ppo_steps = _as_non_negative_int(overrides.get("ppo_total_timesteps", 0))
    eval_episodes = _as_non_negative_int(overrides.get("evaluation_episodes", 0))
    if tool_name == "run_experiment":
        return ppo_steps, eval_episodes
    if tool_name == "run_probe_suite":
        variant_count = 3
        return ppo_steps * variant_count, eval_episodes * variant_count
    return 0, 0


def _as_non_negative_int(raw: object) -> int:
    """
    Convert numeric tool arguments into a non-negative integer estimate.
    """
    if not isinstance(raw, int | float):
        return 0
    return max(0, int(raw))
