from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .launch import PPORunContract
from .launcher import LaunchScriptBuilder


@dataclass(frozen=True)
class ExecutionReceipt:
    """Outcome of a launch attempt or launch preview."""

    mode: str
    status: str
    contract: Dict[str, Any]
    command: str
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PPORunDispatcher:
    """Single entrypoint for planning or dispatching PPO jobs."""

    def preview(self, contract: PPORunContract) -> ExecutionReceipt:
        return ExecutionReceipt(
            mode="preview",
            status="planned",
            contract=contract.to_dict(),
            command=" ".join(contract.render_command()),
            notes="Dispatch not executed locally; contract rendered only.",
            metadata={
                "environment": contract.execution_plan.environment,
                "algorithm": contract.execution_plan.algorithm,
            },
        )

    def render_script(self, contract: PPORunContract, script_path) -> ExecutionReceipt:
        builder = LaunchScriptBuilder()
        script = builder.build(contract, script_path)
        script.write()
        return ExecutionReceipt(
            mode="script",
            status="written",
            contract=contract.to_dict(),
            command=str(script.path),
            notes="Launch script written for future VM execution.",
            metadata={
                "script_path": str(script.path),
                "script_size": len(script.content),
            },
        )

    def dispatch(self, contract: PPORunContract) -> ExecutionReceipt:
        raise NotImplementedError(
            "Remote execution is not wired yet. Use preview() until the GPU VM launcher is added."
        )
