"""
Summary: Unit tests for Gymnasium reward-call argument selection.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import inspect

import pytest

from rewardlab.experiments.execution_service import ExecutionError
from rewardlab.experiments.gymnasium_runner import _select_call_arguments


def _params_from(callable_obj: object) -> tuple[inspect.Parameter, ...]:
    """Return a tuple of signature parameters for helper tests."""

    return tuple(inspect.signature(callable_obj).parameters.values())


def test_select_call_arguments_accepts_var_keyword_without_missing_error() -> None:
    """A reward signature with **kwargs should not fail missing-parameter validation."""

    def reward(observation: object, **kwargs: object) -> float:
        """Return a deterministic scalar while accepting arbitrary keyword context."""

        del kwargs
        return 1.0

    arguments = _select_call_arguments(
        parameters=_params_from(reward),
        available={"observation": [0.0], "x_velocity": 0.4, "terminated": False},
    )

    assert arguments["observation"] == [0.0]
    assert arguments["x_velocity"] == 0.4
    assert arguments["terminated"] is False


def test_select_call_arguments_rejects_positional_only_signatures() -> None:
    """Positional-only reward signatures are not callable via keyword adaptation."""

    def reward(observation: object, /) -> float:
        """Return a constant scalar from a positional-only signature."""

        return 1.0

    with pytest.raises(ExecutionError, match="positional-only"):
        _select_call_arguments(
            parameters=_params_from(reward),
            available={"observation": [0.0]},
        )
