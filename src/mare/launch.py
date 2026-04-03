from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .environment import IsaacGymExecutionPlan
from .manifest import ExperimentManifest


@dataclass(frozen=True)
class LaunchTarget:
    """Where a PPO job is expected to run."""

    kind: str
    python_executable: str
    working_directory: Path
    gpu_required: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["working_directory"] = str(self.working_directory)
        return payload


@dataclass(frozen=True)
class PPORunContract:
    """Structured plan for an IsaacGym PPO job."""

    manifest: ExperimentManifest
    execution_plan: IsaacGymExecutionPlan
    run_dir: Path
    launch_target: LaunchTarget
    entrypoint: str = "scripts/train_ppo.py"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": asdict(self.manifest),
            "execution_plan": self.execution_plan.to_dict(),
            "run_dir": str(self.run_dir),
            "launch_target": self.launch_target.to_dict(),
            "entrypoint": self.entrypoint,
        }

    def render_command(self) -> List[str]:
        command = [
            self.launch_target.python_executable,
            self.entrypoint,
            "--environment",
            self.execution_plan.environment,
            "--task-name",
            self.execution_plan.task_name,
            "--algorithm",
            self.execution_plan.algorithm,
            "--seed",
            str(self.execution_plan.seed),
            "--train-steps",
            str(self.execution_plan.train_steps),
            "--eval-episodes",
            str(self.execution_plan.eval_episodes),
            "--device",
            self.execution_plan.device,
            "--run-dir",
            str(self.run_dir),
        ]
        reward_candidate = self.manifest.reward_candidate or {}
        reward_path = reward_candidate.get("path")
        reward_entrypoint = reward_candidate.get("entrypoint")
        if reward_path:
            command.extend(["--reward-candidate", str(reward_path)])
        if reward_entrypoint:
            command.extend(["--reward-entrypoint", str(reward_entrypoint)])
        return command
