"""
Summary: Integration tests for actual Gymnasium-backed session stepping and persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.gymnasium_runner import GymnasiumExperimentRunner
from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.experiment_run import ExecutionMode, RunStatus
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionStatus


class FakeGymModule:
    """Minimal Gymnasium module double with explicit environment registration."""

    __version__ = "0.29.1"

    def __init__(self, registered_ids: set[str]) -> None:
        """Store the environment ids available to the fake runtime."""

        self.registered_ids = registered_ids

    def spec(self, environment_id: str) -> object:
        """Return a dummy spec object for registered environments."""

        if environment_id not in self.registered_ids:
            raise RuntimeError(f"No registered env with id: {environment_id}")
        return object()


class FakeGymnasiumEnvironment:
    """Deterministic rollout environment for actual-path integration tests."""

    def __init__(self) -> None:
        """Initialize a fixed short rollout."""

        self._steps = [
            ([0.0, 0.0, 0.01, 0.0], 1.0, False, False, {}),
            ([0.0, 0.0, 0.07, 0.0], 1.0, True, False, {}),
        ]
        self._index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Reset the fake rollout and return the initial observation."""

        self._index = 0
        return [0.0, 0.0, 0.0, 0.0], {"seed": seed}

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, int]]:
        """Return the next configured transition."""

        observation, reward, terminated, truncated, info = self._steps[self._index]
        self._index += 1
        return observation, reward, terminated, truncated, {"action": action, **info}

    def close(self) -> None:
        """Track environment closure for cleanup assertions."""

        self.closed = True


def build_actual_service(root: Path) -> SessionService:
    """Create a session service configured for actual Gymnasium execution."""

    backend = GymnasiumBackend(
        environment_factory=lambda **_: FakeGymnasiumEnvironment(),
        gym_module=FakeGymModule({"CartPole-v1"}),
    )
    paths = ServicePaths(
        data_dir=root / ".rewardlab",
        database_path=root / ".rewardlab" / "metadata.sqlite3",
        event_log_dir=root / ".rewardlab" / "events",
        checkpoint_dir=root / ".rewardlab" / "checkpoints",
        report_dir=root / ".rewardlab" / "reports",
    )
    service = SessionService(
        paths=paths,
        gymnasium_runner=GymnasiumExperimentRunner(backend=backend, default_max_episode_steps=5),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    return service


def create_input_files(root: Path) -> tuple[Path, Path]:
    """Create objective and reward-program fixtures for actual-path tests."""

    objective_file = root / "objective.txt"
    objective_file.write_text(
        "Reward stable balance with centered, low-oscillation behavior.",
        encoding="utf-8",
    )
    baseline_reward_file = root / "baseline_reward.py"
    baseline_reward_file.write_text(
        "\n".join(
            [
                "def reward(state):",
                "    return 1.0 if abs(state[2]) < 0.05 else 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def test_actual_gymnasium_step_persists_run_evidence_and_report_summary(
    workspace_tmp_path: Path,
) -> None:
    """An actual Gymnasium step should persist an experiment run and report evidence."""

    service = build_actual_service(workspace_tmp_path)
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="CartPole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-gym",
    )

    stepped = service.step_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)
    stopped = service.stop_session(started.session_id)
    report_payload = json.loads(stopped.report_path.read_text(encoding="utf-8"))

    assert stepped.iteration_index == 1
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETED
    assert runs[0].execution_mode == ExecutionMode.ACTUAL_BACKEND
    assert runs[0].artifact_refs
    matching_iterations = [
        item
        for item in report_payload["iterations"]
        if item["candidate_id"] == stepped.candidate_id
    ]
    assert matching_iterations
    assert runs[0].run_id in matching_iterations[0]["performance_summary"]
    assert "manifest.json" in matching_iterations[0]["performance_summary"]


def test_actual_gymnasium_failure_pauses_session_with_persisted_failed_run(
    workspace_tmp_path: Path,
) -> None:
    """A runtime resolution failure should persist a failed run and pause the session."""

    backend = GymnasiumBackend(gym_module=FakeGymModule(set()))
    paths = ServicePaths(
        data_dir=workspace_tmp_path / ".rewardlab",
        database_path=workspace_tmp_path / ".rewardlab" / "metadata.sqlite3",
        event_log_dir=workspace_tmp_path / ".rewardlab" / "events",
        checkpoint_dir=workspace_tmp_path / ".rewardlab" / "checkpoints",
        report_dir=workspace_tmp_path / ".rewardlab" / "reports",
    )
    service = SessionService(
        paths=paths,
        gymnasium_runner=GymnasiumExperimentRunner(backend=backend, default_max_episode_steps=5),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="MissingEnv-v0",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-gym-failure",
    )

    with pytest.raises(RuntimeError, match="MissingEnv-v0"):
        service.step_session(started.session_id)

    session = service.get_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)

    assert session is not None
    assert session.status == SessionStatus.PAUSED
    assert len(runs) == 1
    assert runs[0].status == RunStatus.FAILED
    assert runs[0].failure_reason is not None
