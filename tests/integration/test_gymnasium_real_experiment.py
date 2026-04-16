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
from rewardlab.experiments.execution_service import ExecutionError, ExecutionRequest
from rewardlab.experiments.gymnasium_runner import (
    GymnasiumExperimentRunner,
    HumanoidPpoEvaluationConfig,
)
from rewardlab.experiments.reward_program import load_reward_program
from rewardlab.llm.openai_client import ChatCompletionRequest, ChatCompletionResponse
from rewardlab.orchestrator.iteration_engine import IterationEngine
from rewardlab.orchestrator.reward_designer import (
    OpenAIRewardDesigner,
    RewardDesignConfig,
    RewardDesignerMode,
)
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


class FakeContinuousActionSpace:
    """Minimal continuous action-space double for fake Humanoid environments."""

    shape = (17,)


class TrainerState:
    """Mutable shared trainer state for fake PPO evaluation tests."""

    def __init__(self, run_index: int) -> None:
        """Store the run index and mutable checkpoint counter."""

        self.run_index = run_index
        self.checkpoint_index = 0


class FakeHumanoidEnvironment:
    """One-step Humanoid-like environment for PPO protocol tests."""

    action_space = FakeContinuousActionSpace()

    def __init__(self, state: TrainerState) -> None:
        """Store the shared state used to report checkpoint progress."""

        self.state = state
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, int | None]]:
        """Return a stable observation vector for both training and evaluation."""

        return [0.0, 0.0, 0.0, 0.0], {"seed": seed}

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict[str, float]]:
        """Terminate immediately and expose a deterministic x_velocity metric."""

        del action
        metric = float(self.state.run_index + self.state.checkpoint_index)
        return [0.0, 0.0, 0.0, 0.0], metric, True, False, {"x_velocity": metric}

    def close(self) -> None:
        """Track environment closure for cleanup assertions."""

        self.closed = True


class FakeHumanoidEnvironmentFactory:
    """Create training and evaluation environments that share per-run state."""

    def __init__(self) -> None:
        """Initialize the pair-tracking state machine."""

        self._shared_state: TrainerState | None = None
        self._calls_in_pair = 0
        self._run_index = 0

    def __call__(
        self,
        *,
        environment_id: str,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> FakeHumanoidEnvironment:
        """Return paired environments with the same shared trainer state."""

        del environment_id, seed, render_mode
        if self._shared_state is None or self._calls_in_pair >= 2:
            self._shared_state = TrainerState(run_index=self._run_index)
            self._run_index += 1
            self._calls_in_pair = 0
        self._calls_in_pair += 1
        return FakeHumanoidEnvironment(self._shared_state)


class FakeTrainer:
    """Minimal PPO trainer double with deterministic checkpoint progress."""

    def __init__(self, state: TrainerState) -> None:
        """Store the shared state used by the fake evaluation environments."""

        self.state = state

    def learn(
        self,
        total_timesteps: int,
        *,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> FakeTrainer:
        """Advance the checkpoint counter once per training slice."""

        del total_timesteps, progress_bar, reset_num_timesteps
        self.state.checkpoint_index += 1
        return self

    def predict(self, observation, deterministic: bool = True) -> tuple[list[float], None]:
        """Return a zero action compatible with the fake Humanoid action space."""

        del observation, deterministic
        return [0.0] * 17, None


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


def test_humanoid_ppo_protocol_reports_paper_style_checkpoint_metrics() -> None:
    """Humanoid execution should aggregate the best checkpoint metric across PPO runs."""

    environment_factory = FakeHumanoidEnvironmentFactory()
    backend = GymnasiumBackend(
        environment_factory=environment_factory,
        gym_module=FakeGymModule({"Humanoid-v4"}),
    )
    runner = GymnasiumExperimentRunner(
        backend=backend,
        humanoid_ppo_config=HumanoidPpoEvaluationConfig(
            total_timesteps=300,
            checkpoint_count=3,
            evaluation_run_count=2,
            evaluation_episodes_per_checkpoint=1,
        ),
        ppo_trainer_factory=lambda environment, seed, config: FakeTrainer(
            environment.environment.state
        ),
    )
    reward_program = load_reward_program(
        candidate_id="candidate-humanoid",
        source_text="def reward(observation, x_velocity):\n    return float(x_velocity)\n",
    )

    outcome = runner(
        ExecutionRequest(
            run_id="run-humanoid",
            backend=EnvironmentBackend.GYMNASIUM,
            environment_id="Humanoid-v4",
            execution_mode=ExecutionMode.ACTUAL_BACKEND,
            seed=11,
        ),
        reward_program,
    )

    assert outcome.metrics["evaluation_protocol"] == "humanoid_ppo_max_checkpoint_mean_x_velocity"
    assert outcome.metrics["per_run_best_mean_x_velocity"] == [3.0, 4.0]
    assert outcome.metrics["fitness_metric_mean"] == pytest.approx(3.5)
    assert len(outcome.event_trace or []) == 6


def test_humanoid_ppo_protocol_reports_missing_sb3_prerequisite(monkeypatch) -> None:
    """Humanoid PPO should fail with an actionable prerequisite when SB3 is unavailable."""

    backend = GymnasiumBackend(
        environment_factory=lambda **_: FakeHumanoidEnvironment(TrainerState(run_index=0)),
        gym_module=FakeGymModule({"Humanoid-v4"}),
    )
    runner = GymnasiumExperimentRunner(backend=backend)
    reward_program = load_reward_program(
        candidate_id="candidate-humanoid",
        source_text="def reward(observation, x_velocity):\n    return float(x_velocity)\n",
    )

    monkeypatch.setattr(
        "rewardlab.experiments.gymnasium_runner._default_ppo_trainer_factory",
        lambda: None,
    )

    with pytest.raises(ExecutionError, match="stable_baselines3"):
        runner(
            ExecutionRequest(
                run_id="run-humanoid-missing-ppo",
                backend=EnvironmentBackend.GYMNASIUM,
                environment_id="Humanoid-v4",
                execution_mode=ExecutionMode.ACTUAL_BACKEND,
            ),
            reward_program,
        )


def test_actual_gymnasium_step_uses_model_backed_reward_designer(
    workspace_tmp_path: Path,
) -> None:
    """Actual backend stepping should use the model-backed designer when configured."""

    captured_request: ChatCompletionRequest | None = None
    generated_reward = (
        "def reward(state, env_reward, terminated):\n"
        "    if terminated:\n"
        "        return -5.0\n"
        "    return float(env_reward + 0.25)\n"
    )

    class FakeOpenAIClient:
        """Capture the outgoing reward-design request and return valid reward code."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Return a stable JSON payload for one reward-design iteration."""

            nonlocal captured_request
            captured_request = request
            return ChatCompletionResponse(
                content=(
                    '{"reward_definition": '
                    '"def reward(state, env_reward, terminated):\\n    if terminated:\\n'
                    '        return -5.0\\n    return float(env_reward + 0.25)\\n", '
                    '"change_summary": "Use environment reward directly with a small bonus.", '
                    '"proposed_changes": ["Increase the shaped reward by a constant margin."]}'
                ),
                raw_response=None,
            )

    backend = GymnasiumBackend(
        environment_factory=lambda **_: FakeGymnasiumEnvironment(),
        gym_module=FakeGymModule({"CartPole-v1"}),
    )
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
        iteration_engine=IterationEngine(
            reward_designer=OpenAIRewardDesigner(
                openai_client=FakeOpenAIClient(),
                config=RewardDesignConfig(
                    mode=RewardDesignerMode.OPENAI,
                    model="gpt-5-nano",
                    reasoning_effort="low",
                    max_tokens=700,
                ),
            )
        ),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="CartPole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-gym-openai",
    )

    stepped = service.step_session(started.session_id)
    candidates = service.list_candidates(started.session_id)
    session = service.get_session(started.session_id)

    assert captured_request is not None
    assert captured_request.model == "gpt-5-nano"
    assert "Reward stable balance" in captured_request.messages[1].content
    assert candidates[-1].candidate_id == stepped.candidate_id
    assert candidates[-1].reward_definition == generated_reward.strip()
    assert session is not None
    assert session.metadata["reward_designer_mode"] == "openai"
    assert session.metadata["reward_designer_model"] == "gpt-5-nano"


def test_actual_gymnasium_design_failure_pauses_before_execution(
    workspace_tmp_path: Path,
) -> None:
    """A reward-design failure should pause the session before execution starts."""

    class FakeOpenAIClient:
        """Return invalid callable parameters so candidate generation fails."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Return an invalid reward signature for failure-path coverage."""

            del request
            return ChatCompletionResponse(
                content=(
                    '{"reward_definition": '
                    '"def reward(forbidden_signal):\\n    return 1.0\\n", '
                    '"change_summary": "Invalid reward.", '
                    '"proposed_changes": ["Use a forbidden signal."]}'
                ),
                raw_response=None,
            )

    backend = GymnasiumBackend(
        environment_factory=lambda **_: FakeGymnasiumEnvironment(),
        gym_module=FakeGymModule({"CartPole-v1"}),
    )
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
        iteration_engine=IterationEngine(
            reward_designer=OpenAIRewardDesigner(
                openai_client=FakeOpenAIClient(),
                config=RewardDesignConfig(mode=RewardDesignerMode.OPENAI),
            )
        ),
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
    )
    service.initialize()
    objective_file, baseline_reward_file = create_input_files(workspace_tmp_path)
    started = service.start_session(
        objective_file=objective_file,
        baseline_reward_file=baseline_reward_file,
        environment_id="CartPole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        session_id="session-actual-gym-openai-failure",
    )

    with pytest.raises(RuntimeError, match="unsupported callable parameters"):
        service.step_session(started.session_id)

    session = service.get_session(started.session_id)
    runs = service.list_experiment_runs(session_id=started.session_id)

    assert session is not None
    assert session.status == SessionStatus.PAUSED
    assert session.metadata["last_failed_design_error"].startswith(
        "reward designer introduced unsupported callable parameters"
    )
    assert runs == []
