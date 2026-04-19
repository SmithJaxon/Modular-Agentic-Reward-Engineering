"""
Summary: Regression tests for Isaac Gym subprocess execution helpers.
Created: 2026-04-19
Last Updated: 2026-04-19
"""

from __future__ import annotations

import pytest

from rewardlab.experiments import isaacgym_runner as runner
from rewardlab.experiments.execution_service import ExecutionError, ExecutionRequest
from rewardlab.experiments.reward_program import (
    SIGNATURE_VERSION,
    RewardProgram,
    RewardProgramStatus,
)
from rewardlab.schemas.experiment_run import ExecutionMode, RunType
from rewardlab.schemas.session_config import EnvironmentBackend


class _BackendStub:
    """Minimal backend stub carrying only the config override used by the runner."""

    _cfg_dir_override = None


class _TemporaryDirectoryStub:
    """Workspace-local TemporaryDirectory replacement for sandbox-safe tests."""

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        del exc_type, exc, tb
        return False


def _request() -> ExecutionRequest:
    """Return a minimal Isaac execution request for subprocess runner tests."""

    return ExecutionRequest(
        run_id="run-timeout",
        backend=EnvironmentBackend.ISAAC_GYM,
        environment_id="Cartpole",
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        seed=7,
        entrypoint_name="reward",
    )


def _reward_program() -> RewardProgram:
    """Return a minimal valid reward program container for subprocess tests."""

    return RewardProgram(
        candidate_id="candidate-timeout",
        entrypoint_name="reward",
        source_text="def reward(**kwargs): return 0.0",
        signature_version=SIGNATURE_VERSION,
        validation_status=RewardProgramStatus.VALID,
    )


def test_tail_lines_decodes_bytes_payloads() -> None:
    """Tail extraction should decode byte streams before joining lines."""

    assert runner._tail_lines(b"alpha\nbeta\ngamma", 2) == "beta\ngamma"


def test_run_isolated_subprocess_surfaces_timeout_bytes(
    monkeypatch,
    workspace_tmp_path,
) -> None:
    """Timeout handling should decode byte stdout/stderr instead of raising TypeError."""

    def _raise_timeout(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise runner.subprocess.TimeoutExpired(
            cmd=["python", "worker.py"],
            timeout=31,
            output=b"booting\nstill running",
            stderr=b"warn one\nwarn two",
        )

    monkeypatch.setattr(runner.subprocess, "run", _raise_timeout)
    monkeypatch.setattr(
        runner.tempfile,
        "TemporaryDirectory",
        lambda prefix="rewardlab_isaac_run_": _TemporaryDirectoryStub(
            workspace_tmp_path / f"{prefix}timeout"
        ),
    )

    with pytest.raises(ExecutionError, match="Isaac Gym isolated run timed out") as exc_info:
        runner._run_isolated_subprocess(
            execution_request=_request(),
            reward_program=_reward_program(),
            policy_config=runner.IsaacGymPolicyConfig(),
            subprocess_config=runner.IsaacGymSubprocessConfig(timeout_seconds=31),
            backend=_BackendStub(),
        )

    message = str(exc_info.value)
    assert "still running" in message
    assert "warn two" in message
