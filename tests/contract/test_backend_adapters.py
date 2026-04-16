"""
Summary: Contract tests for the Gymnasium backend adapter.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.experiments.backends.base import BackendEpisodeResult
from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend


class FakeEnvironment:
    """Minimal environment double for backend adapter contract tests."""

    def __init__(self, rewards: list[float]) -> None:
        """Store a deterministic reward sequence for later rollout."""

        self._rewards = list(rewards)
        self._index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, int | None]]:
        """Reset the fake environment and return a simple observation."""

        self._index = 0
        return 0, {"seed": seed}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
        """Return the next deterministic reward and terminate at the final step."""

        reward = self._rewards[self._index]
        self._index += 1
        terminated = self._index >= len(self._rewards)
        return self._index, reward, terminated, False, {"action": action}

    def close(self) -> None:
        """Track whether the adapter closed the environment handle."""

        self.closed = True


def test_gymnasium_backend_runs_episode_with_injected_factory() -> None:
    """The Gymnasium adapter should support rollouts through the shared interface."""

    environment = FakeEnvironment([1.0, 2.0, 3.0])
    backend = GymnasiumBackend(environment_factory=lambda **_: environment)

    result = backend.run_episode("cartpole-v1", policy=lambda _: 0, seed=7)

    assert isinstance(result, BackendEpisodeResult)
    assert result.total_reward == 6.0
    assert result.metadata["backend"] == "gymnasium"
    assert environment.closed is True
