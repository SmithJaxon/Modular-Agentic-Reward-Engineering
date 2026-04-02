"""
Summary: Shared backend adapter interface for RewardLab experiment execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class BackendStepResult:
    """Represents one step of backend execution during an experiment rollout."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendEpisodeResult:
    """Represents the outcome of a backend rollout for later orchestration use."""

    steps: tuple[BackendStepResult, ...]
    total_reward: float
    terminal_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BackendEnvironment(Protocol):
    """Protocol for environment handles returned by backend adapters."""

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one action."""

    def close(self) -> None:
        """Release any backend resources."""


class BackendAdapter(ABC):
    """Abstract base class for environment backend adapters."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the stable backend identifier used in session configuration."""

    @abstractmethod
    def supports(self, environment_id: str) -> bool:
        """Return whether this adapter can execute the requested environment."""

    @abstractmethod
    def create_environment(
        self,
        environment_id: str,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BackendEnvironment:
        """Create a backend-specific environment instance for the requested id."""

    def close_environment(self, environment: BackendEnvironment) -> None:
        """Close an environment handle if the backend exposes lifecycle cleanup."""

        environment.close()

    def run_episode(
        self,
        environment_id: str,
        policy: Any,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BackendEpisodeResult:
        """Run a single backend rollout using the supplied policy callable.

        The policy is expected to accept the latest observation and return an
        action compatible with the backend environment. This method is kept
        intentionally small so concrete adapters can specialize it later while
        tests can rely on a consistent contract.
        """

        environment = self.create_environment(
            environment_id,
            seed=seed,
            render_mode=render_mode,
        )
        steps: list[BackendStepResult] = []
        total_reward = 0.0
        terminal_reason: str | None = None

        try:
            observation, _ = environment.reset(seed=seed)
            while True:
                action = policy(observation)
                observation, reward, terminated, truncated, info = environment.step(action)
                step_result = BackendStepResult(
                    observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=dict(info),
                )
                steps.append(step_result)
                total_reward += reward
                if terminated:
                    terminal_reason = "terminated"
                    break
                if truncated:
                    terminal_reason = "truncated"
                    break
        finally:
            self.close_environment(environment)

        return BackendEpisodeResult(
            steps=tuple(steps),
            total_reward=total_reward,
            terminal_reason=terminal_reason,
            metadata={"environment_id": environment_id, "backend": self.backend_name},
        )
