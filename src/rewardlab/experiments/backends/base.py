"""
Summary: Base backend adapter contracts for experiment execution backends.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rewardlab.schemas.session_config import EnvironmentBackend


@dataclass(slots=True, frozen=True)
class ExperimentInput:
    """
    Represent normalized experiment execution input values.
    """

    session_id: str
    environment_id: str
    environment_backend: EnvironmentBackend
    reward_definition: str
    iteration_index: int
    objective_text: str
    variant_label: str = "default"
    seed: int = 0
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ExperimentOutput:
    """
    Represent normalized backend experiment output values.
    """

    score: float
    metrics: dict[str, Any]
    summary: str
    artifact_refs: tuple[str, ...] = ()


class EnvironmentBackendAdapter(ABC):
    """
    Define the backend adapter contract for experiment execution.
    """

    @abstractmethod
    def run_performance(self, payload: ExperimentInput) -> ExperimentOutput:
        """
        Execute performance experiment for a reward candidate.
        """

    @abstractmethod
    def run_reflection(self, payload: ExperimentInput) -> ExperimentOutput:
        """
        Execute reflection experiment for a reward candidate.
        """
