"""
Summary: Integration test for actual run artifacts attached to feedback flows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.experiments.backends.gymnasium_backend import GymnasiumBackend
from rewardlab.experiments.gymnasium_runner import GymnasiumExperimentRunner
from rewardlab.orchestrator.session_service import ServicePaths, SessionService
from rewardlab.schemas.experiment_run import ExecutionMode
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
    """Deterministic rollout environment for actual-path feedback tests."""

    def __init__(self) -> None:
        """Initialize a short stable rollout."""

        self._steps = [
            ([0.1, 0.0, 0.0, 0.0], 1.0, False, False, {}),
            ([0.2, 0.0, 0.0, 0.0], 1.0, True, False, {}),
        ]
        self._index = 0

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Reset the rollout and return the initial observation."""

        self._index = 0
        return [0.0, 0.0, 0.0, 0.0], {"seed": seed}

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, int]]:
        """Return the next configured transition."""

        observation, reward, terminated, truncated, info = self._steps[self._index]
        self._index += 1
        return observation, reward, terminated, truncated, {"action": action, **info}

    def close(self) -> None:
        """Close the fake environment handle."""


def test_feedback_defaults_to_latest_actual_run_artifact(
    workspace_tmp_path: Path,
) -> None:
    """Human and peer feedback should point at the latest actual run artifact bundle."""

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
        session_id="session-actual-feedback-artifacts",
    )

    stepped = service.step_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)
    manifest_ref = next(
        artifact_ref
        for artifact_ref in runs[0].artifact_refs
        if artifact_ref.endswith("manifest.json")
    )

    human_feedback = service.submit_human_feedback(
        session_id=started.session_id,
        candidate_id=stepped.candidate_id,
        comment="The rollout stayed stable and reviewable.",
        score=0.9,
    )
    peer_feedback = service.request_peer_feedback(
        session_id=started.session_id,
        candidate_id=stepped.candidate_id,
    )

    review_bundle_root = (
        workspace_tmp_path
        / ".rewardlab"
        / "reports"
        / "feedback_artifacts"
        / human_feedback.feedback_id
    )
    review_manifest = json.loads(
        (review_bundle_root / "manifest.json").read_text(encoding="utf-8")
    )
    review_markdown = (review_bundle_root / "review.md").read_text(encoding="utf-8")

    assert human_feedback.artifact_ref == manifest_ref
    assert peer_feedback.artifact_ref == manifest_ref
    assert review_manifest["artifact_ref"] == manifest_ref
    assert manifest_ref in review_markdown


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
    """Create objective and reward-program fixtures for feedback artifact tests."""

    objective_file = root / "objective.txt"
    objective_file.write_text(
        "Reward steady positive cart movement with low oscillation.",
        encoding="utf-8",
    )
    baseline_reward_file = root / "baseline_reward.py"
    baseline_reward_file.write_text(
        "\n".join(
            [
                "def reward(cart_position):",
                "    return cart_position + 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return objective_file, baseline_reward_file
