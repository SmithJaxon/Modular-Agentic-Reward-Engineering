from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class IsaacGymExecutionPlan:
    """Execution contract for an IsaacGym-backed run."""

    environment: str
    task_name: str
    algorithm: str
    seed: int
    train_steps: int
    eval_episodes: int
    device: str = "cuda"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IsaacGymEnvironmentProfile:
    """Static metadata for a target environment."""

    name: str
    task_name: str
    default_train_steps: int
    default_eval_episodes: int
    default_device: str = "cuda"
    default_hyperparameters: Dict[str, Any] = field(default_factory=dict)
