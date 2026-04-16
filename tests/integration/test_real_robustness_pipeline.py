"""
Summary: Integration test for actual-backend robustness execution and persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.execution_service import ExperimentExecutionService
from rewardlab.experiments.gymnasium_runner import GymnasiumExperimentRunner
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.experiment_run import ExecutionMode, RunType
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate


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
    """Deterministic rollout environment that exposes configured observations."""

    def __init__(self, observations: list[list[float]]) -> None:
        """Store the sequence of observations used for the rollout."""

        self._observations = observations
        self._index = 0

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Reset the rollout and return the initial observation."""

        self._index = 0
        return [0.0, 0.0, 0.0, 0.0], {"seed": seed}

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, int]]:
        """Return the next configured transition."""

        observation = self._observations[self._index]
        self._index += 1
        terminated = self._index >= len(self._observations)
        return observation, 1.0, terminated, False, {"action": action}

    def close(self) -> None:
        """Close the fake environment handle."""


def test_actual_step_persists_robustness_runs_and_assessment(
    workspace_tmp_path: Path,
) -> None:
    """Actual backend stepping should persist robustness probe evidence and assessments."""

    probe_matrix_path = write_probe_matrix(workspace_tmp_path)
    service = build_service(workspace_tmp_path, probe_matrix_path)
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="CartPole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-robustness",
    )

    stepped = service.step_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)
    assessments = service.list_robustness_assessments(session_id=started.session_id)
    stopped = service.stop_session(started.session_id)
    report_payload = json.loads(stopped.report_path.read_text(encoding="utf-8"))

    performance_runs = [run for run in runs if run.run_type == RunType.PERFORMANCE]
    robustness_runs = [run for run in runs if run.run_type == RunType.ROBUSTNESS]

    assert len(performance_runs) == 1
    assert len(robustness_runs) == 2
    assert all(run.execution_mode == ExecutionMode.ACTUAL_BACKEND for run in robustness_runs)
    assert all(run.artifact_refs for run in robustness_runs)
    assert len(assessments) == 1
    assert assessments[0].candidate_id == stepped.candidate_id
    assert assessments[0].primary_run_id == performance_runs[0].run_id
    assert assessments[0].probe_run_ids == [run.run_id for run in robustness_runs]
    assert assessments[0].backend == EnvironmentBackend.GYMNASIUM
    assert assessments[0].risk_level.value == "high"

    matching_iterations = [
        item
        for item in report_payload["iterations"]
        if item["candidate_id"] == stepped.candidate_id
    ]
    assert matching_iterations
    assert matching_iterations[0]["risk_level"] == "high"
    assert "robustness risk high" in matching_iterations[0]["performance_summary"].lower()


def build_service(root: Path, probe_matrix_path: Path) -> SessionService:
    """Create a session service configured for actual robustness execution."""

    observations_by_environment = {
        "CartPole-v1": [
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        "CartPole-v1-noisy": [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        "CartPole-v1-shifted": [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
    }
    backend = GymnasiumBackend(
        environment_factory=lambda environment_id, **_: FakeGymnasiumEnvironment(
            observations_by_environment[environment_id]
        ),
        gym_module=FakeGymModule(set(observations_by_environment)),
    )
    paths = ServicePaths(
        data_dir=root / ".rewardlab",
        database_path=root / ".rewardlab" / "metadata.sqlite3",
        event_log_dir=root / ".rewardlab" / "events",
        checkpoint_dir=root / ".rewardlab" / "checkpoints",
        report_dir=root / ".rewardlab" / "reports",
    )
    execution_service = ExperimentExecutionService(paths.experiment_artifact_writer())
    gymnasium_runner = GymnasiumExperimentRunner(
        backend=backend,
        default_max_episode_steps=5,
    )
    service = SessionService(
        paths=paths,
        experiment_execution_service=execution_service,
        gymnasium_runner=gymnasium_runner,
        robustness_runner=RobustnessRunner(
            probe_matrix_path=probe_matrix_path,
            experiment_execution_service=execution_service,
            gymnasium_runner=gymnasium_runner,
        ),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    return service


def create_input_files(root: Path) -> tuple[Path, Path]:
    """Create objective and reward-program fixtures for robustness tests."""

    objective_file = root / "objective.txt"
    objective_file.write_text(
        "Reward strong positive cart position while remaining stable.",
        encoding="utf-8",
    )
    baseline_reward_file = root / "baseline_reward.py"
    baseline_reward_file.write_text(
        "\n".join(
            [
                "def reward(cart_position):",
                "    return float(cart_position)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file


def write_probe_matrix(root: Path) -> Path:
    """Write a small probe matrix aligned with the fake environments."""

    probe_matrix_path = root / "probe_matrix.json"
    probe_matrix_path.write_text(
        json.dumps(
            {
                "version": 1,
                "backends": {
                    "gymnasium": [
                        {
                            "label": "cartpole-v1-noisy",
                            "environment_id": "CartPole-v1-noisy",
                            "seed": 7,
                            "overrides": {"action_noise": 0.01},
                        },
                        {
                            "label": "cartpole-v1-shifted",
                            "environment_id": "CartPole-v1-shifted",
                            "seed": 13,
                            "overrides": {"reward_scale": 0.8},
                        },
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return probe_matrix_path
