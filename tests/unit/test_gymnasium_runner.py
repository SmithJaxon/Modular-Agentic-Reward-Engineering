"""
Summary: Unit tests for Gymnasium reward scoring helpers.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from fractions import Fraction

from rewardlab.experiments.gymnasium_runner import _build_reward_arguments


class ArrayLikeObservation:
    """Minimal array-like observation double with numeric indexing semantics."""

    def __init__(self, values: list[float]) -> None:
        """Store the underlying numeric values."""

        self._values = values

    def __len__(self) -> int:
        """Return the number of values exposed by the observation."""

        return len(self._values)

    def __getitem__(self, index: int) -> float:
        """Return the value at the requested observation index."""

        return self._values[index]


def test_build_reward_arguments_supports_array_like_observations() -> None:
    """Array-like observations should expose CartPole-style named arguments."""

    arguments = _build_reward_arguments(
        observation=ArrayLikeObservation([0.1, 0.2, 0.3, 0.4]),
        env_reward=1.0,
        terminated=False,
        truncated=False,
        action=1,
        step_index=2,
        info={"source": "unit"},
    )

    assert arguments["cart_position"] == 0.1
    assert arguments["cart_velocity"] == 0.2
    assert arguments["pole_angle_radians"] == 0.3
    assert arguments["angular_velocity"] == 0.4
    assert arguments["state"] is not None


def test_build_reward_arguments_supports_non_builtin_real_scalars() -> None:
    """Array-like observations should accept real-number scalars beyond float/int."""

    arguments = _build_reward_arguments(
        observation=ArrayLikeObservation(
            [
                Fraction(1, 10),
                Fraction(2, 10),
                Fraction(3, 10),
                Fraction(4, 10),
            ]
        ),
        env_reward=1.0,
        terminated=False,
        truncated=False,
        action=1,
        step_index=2,
        info={"source": "unit"},
    )

    assert arguments["cart_position"] == 0.1
    assert arguments["cart_velocity"] == 0.2
    assert arguments["pole_angle_radians"] == 0.3
    assert arguments["angular_velocity"] == 0.4
