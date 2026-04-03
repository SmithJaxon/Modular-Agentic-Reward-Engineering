"""
Summary: Integration tests for actual Isaac-backed session stepping and persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rewardlab.experiments.backends.isaacgym_backend import IsaacGymBackend
from rewardlab.experiments.isaacgym_runner import IsaacGymExperimentRunner
from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.experiment_run import ExecutionMode, RunStatus
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionStatus


class FakeIsaacModule:
    """Minimal Isaac module double exposing a version string for readiness checks."""

    __version__ = "preview-0.1"


class FakeIsaacEnvironment:
    """Deterministic rollout environment for actual-path Isaac integration tests."""

    def __init__(
        self,
        transitions: list[tuple[list[float], float, bool, bool, dict[str, str]]],
    ) -> None:
        """Store the transition sequence used during the rollout."""

        self._transitions = transitions
        self._index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Reset the fake rollout and return the initial observation."""

        self._index = 0
        return [0.0, 0.0], {"seed": seed}

    def default_action(self) -> list[float]:
        """Return a deterministic zero-action vector for the rollout."""

        return [0.0, 0.0]

    def step(
        self,
        action: list[float],
    ) -> tuple[list[float], float, bool, bool, dict[str, str | list[float]]]:
        """Return the next configured transition."""

        observation, reward, terminated, truncated, info = self._transitions[self._index]
        self._index += 1
        return observation, reward, terminated, truncated, {"action": action, **info}

    def close(self) -> None:
        """Track environment closure for cleanup assertions."""

        self.closed = True


def test_actual_isaac_step_persists_run_evidence_and_report_summary(
    workspace_tmp_path: Path,
) -> None:
    """An actual Isaac step should persist an experiment run and report evidence."""

    transitions_by_environment = {
        "Isaac-Cartpole-v0": [
            ([0.2, 0.0], 1.0, False, False, {"phase": "stabilize"}),
            ([0.4, 0.1], 1.0, True, False, {"phase": "finish"}),
        ]
    }
    backend = IsaacGymBackend(
        environment_factory=lambda environment_id, **_: FakeIsaacEnvironment(
            transitions_by_environment[environment_id]
        ),
        isaac_module=FakeIsaacModule(),
        environment_validator=_validator_for(set(transitions_by_environment)),
    )
    service = build_actual_service(workspace_tmp_path, backend=backend)
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="Isaac-Cartpole-v0",
        environment_backend=EnvironmentBackend.ISAACGYM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-isaac",
    )

    stepped = service.step_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)
    stopped = service.stop_session(started.session_id)
    report_payload = json.loads(stopped.report_path.read_text(encoding="utf-8"))

    assert stepped.iteration_index == 1
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETED
    assert runs[0].execution_mode == ExecutionMode.ACTUAL_BACKEND
    assert runs[0].backend == EnvironmentBackend.ISAACGYM
    assert runs[0].artifact_refs
    matching_iterations = [
        item
        for item in report_payload["iterations"]
        if item["candidate_id"] == stepped.candidate_id
    ]
    assert matching_iterations
    assert runs[0].run_id in matching_iterations[0]["performance_summary"]
    assert "Isaac-Cartpole-v0" in matching_iterations[0]["performance_summary"]


def test_actual_isaac_missing_factory_pauses_session_with_persisted_failed_run(
    workspace_tmp_path: Path,
) -> None:
    """A missing Isaac factory configuration should pause the session with a failed run."""

    backend = IsaacGymBackend(isaac_module=FakeIsaacModule())
    service = build_actual_service(workspace_tmp_path, backend=backend)
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="Isaac-Cartpole-v0",
        environment_backend=EnvironmentBackend.ISAACGYM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-isaac-failure",
    )

    with pytest.raises(RuntimeError, match="REWARDLAB_ISAAC_ENV_FACTORY"):
        service.step_session(started.session_id)

    session = service.get_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)

    assert session is not None
    assert session.status == SessionStatus.PAUSED
    assert len(runs) == 1
    assert runs[0].status == RunStatus.FAILED
    assert runs[0].failure_reason is not None
    assert "REWARDLAB_ISAAC_ENV_FACTORY" in runs[0].failure_reason


def build_actual_service(root: Path, *, backend: IsaacGymBackend) -> SessionService:
    """Create a session service configured for actual Isaac execution."""

    paths = ServicePaths(
        data_dir=root / ".rewardlab",
        database_path=root / ".rewardlab" / "metadata.sqlite3",
        event_log_dir=root / ".rewardlab" / "events",
        checkpoint_dir=root / ".rewardlab" / "checkpoints",
        report_dir=root / ".rewardlab" / "reports",
    )
    service = SessionService(
        paths=paths,
        isaacgym_runner=IsaacGymExperimentRunner(backend=backend, default_max_episode_steps=5),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    return service


def create_input_files(root: Path) -> tuple[Path, Path]:
    """Create objective and reward-program fixtures for actual Isaac-path tests."""

    objective_file = root / "objective.txt"
    objective_file.write_text(
        "Reward smooth positive state progress with stable Isaac dynamics.",
        encoding="utf-8",
    )
    baseline_reward_file = root / "baseline_reward.py"
    baseline_reward_file.write_text(
        "\n".join(
            [
                "def reward(state, environment_reward):",
                "    return float(state[0]) + float(environment_reward)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def _validator_for(environment_ids: set[str]) -> Callable[[str], None]:
    """Return a validator that only accepts the provided environment ids."""

    def validate(environment_id: str) -> None:
        """Raise when the requested environment is not supported."""

        if environment_id not in environment_ids:
            raise RuntimeError(f"Unsupported Isaac environment: {environment_id}")

    return validate
