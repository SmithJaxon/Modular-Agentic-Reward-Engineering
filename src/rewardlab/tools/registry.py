"""
Summary: Tool manifest registry for modular agentic tool execution.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rewardlab.schemas.budget_state import BudgetState
from rewardlab.schemas.tool_contracts import ToolRequest, ToolResult

ToolExecutor = Callable[[ToolRequest, BudgetState], ToolResult]


@dataclass(slots=True, frozen=True)
class ToolManifest:
    """
    Describe one registered tool capability.
    """

    name: str
    version: str
    description: str
    executor: ToolExecutor


class ToolRegistry:
    """
    Hold registered tool manifests and resolve executors by name.
    """

    def __init__(self) -> None:
        """
        Initialize an empty tool registry.
        """
        self._manifests: dict[str, ToolManifest] = {}

    def register(
        self,
        *,
        name: str,
        version: str,
        description: str,
        executor: ToolExecutor,
    ) -> None:
        """
        Register or replace one tool manifest in the registry.
        """
        self._manifests[name] = ToolManifest(
            name=name,
            version=version,
            description=description,
            executor=executor,
        )

    def get(self, name: str) -> ToolManifest | None:
        """
        Resolve one manifest by name.
        """
        return self._manifests.get(name)

    def list_names(self) -> tuple[str, ...]:
        """
        Return a stable tuple of registered tool names.
        """
        return tuple(sorted(self._manifests))
