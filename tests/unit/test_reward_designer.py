"""
Summary: Unit tests for deterministic and OpenAI-backed reward design strategies.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rewardlab.llm.openai_client import ChatCompletionRequest, ChatCompletionResponse
from rewardlab.orchestrator.reward_designer import (
    OpenAIRewardDesigner,
    RewardDesignConfig,
    RewardDesignerMode,
    RewardDesignRequest,
    resolve_reward_designer,
)
from rewardlab.schemas.experiment_run import ExecutionMode, ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import EnvironmentBackend


def build_candidate(**overrides: object) -> RewardCandidate:
    """Create a valid reward candidate for reward-designer tests."""

    payload: dict[str, object] = {
        "candidate_id": "candidate-001",
        "session_id": "session-001",
        "iteration_index": 1,
        "reward_definition": (
            "def reward(observation, x_velocity, reward_alive, reward_quadctrl, terminated):\n"
            "    if terminated:\n"
            "        return -5.0\n"
            "    return float(x_velocity + reward_alive - reward_quadctrl)\n"
        ),
        "change_summary": "Current baseline reward for Humanoid.",
        "aggregate_score": 0.48,
    }
    payload.update(overrides)
    return RewardCandidate.model_validate(payload)


def build_reflection() -> ReflectionRecord:
    """Create a reflection record for prompt-construction tests."""

    return ReflectionRecord(
        reflection_id="reflection-001",
        candidate_id="candidate-001",
        source_run_ids=["run-001"],
        summary="Forward speed improved, but torso stability is still weak.",
        proposed_changes=[
            "Increase upright posture incentives.",
            "Reduce collapse-prone observation penalties.",
        ],
        confidence=0.73,
    )


def build_run() -> ExperimentRun:
    """Create a completed Humanoid experiment run for prompt-construction tests."""

    started_at = datetime(2026, 4, 6, 18, 0, tzinfo=UTC)
    ended_at = datetime(2026, 4, 6, 18, 8, tzinfo=UTC)
    return ExperimentRun(
        run_id="run-001",
        candidate_id="candidate-001",
        backend=EnvironmentBackend.GYMNASIUM,
        environment_id="Humanoid-v4",
        run_type=RunType.PERFORMANCE,
        execution_mode=ExecutionMode.ACTUAL_BACKEND,
        variant_label="default",
        status=RunStatus.COMPLETED,
        metrics={
            "fitness_metric_name": "mean_x_velocity",
            "fitness_metric_mean": 0.482898,
            "per_run_best_mean_x_velocity": [0.57, 0.44, 0.51, 0.53, 0.34],
        },
        artifact_refs=["artifacts/run-001/manifest.json"],
        started_at=started_at,
        ended_at=ended_at,
    )


def build_request(**overrides: object) -> RewardDesignRequest:
    """Create a valid reward-design request for model-backed tests."""

    payload: dict[str, object] = {
        "session_id": "session-001",
        "objective_text": "Make the humanoid run forward quickly with an upright, stable gait.",
        "environment_id": "Humanoid-v4",
        "environment_backend": EnvironmentBackend.GYMNASIUM,
        "current_candidate": build_candidate(),
        "next_iteration_index": 2,
        "latest_reflection": build_reflection(),
        "latest_run": build_run(),
    }
    payload.update(overrides)
    return RewardDesignRequest(**payload)


def test_resolve_reward_designer_defaults_to_deterministic_mode() -> None:
    """Reward design should stay offline-safe unless OpenAI mode is explicitly requested."""

    designer = resolve_reward_designer(
        config=RewardDesignConfig(mode=RewardDesignerMode.DETERMINISTIC)
    )

    assert designer.mode == RewardDesignerMode.DETERMINISTIC


def test_openai_reward_designer_builds_valid_json_request_and_response() -> None:
    """The model-backed designer should build the request and parse valid JSON output."""

    captured_request: ChatCompletionRequest | None = None

    class FakeOpenAIClient:
        """Fake model client used to capture the outgoing reward-design request."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Capture the request and return a valid reward-design JSON payload."""

            nonlocal captured_request
            captured_request = request
            return ChatCompletionResponse(
                content=(
                    '{"reward_definition": '
                    '"def reward(observation, x_velocity, reward_alive, reward_quadctrl, '
                    'terminated):\\n    if terminated:\\n        return -10.0\\n    '
                    'upright_bonus = 0.1 if observation[0] > 1.0 else -0.2\\n    '
                    'return float((1.5 * x_velocity) + reward_alive - reward_quadctrl + '
                    'upright_bonus)\\n", '
                    '"change_summary": "Increase forward-velocity weight and add a simple '
                    'upright bonus.", '
                    '"proposed_changes": ["Scale forward speed more aggressively.", '
                    '"Add a small upright posture bonus from observation[0]."]}'
                ),
                raw_response=None,
            )

    designer = OpenAIRewardDesigner(
        openai_client=FakeOpenAIClient(),
        config=RewardDesignConfig(
            mode=RewardDesignerMode.OPENAI,
            model="gpt-5-nano",
            reasoning_effort="low",
            max_tokens=800,
        ),
    )

    result = designer.design_next_candidate(build_request())

    assert captured_request is not None
    assert captured_request.model == "gpt-5-nano"
    assert captured_request.reasoning_effort == "low"
    assert captured_request.max_tokens == 800
    assert captured_request.response_format == {"type": "json_object"}
    assert "Forward speed improved" in captured_request.messages[1].content
    assert "fitness_metric_mean" in captured_request.messages[1].content
    assert "x_velocity" in captured_request.messages[1].content
    assert result.change_summary.startswith("Increase forward-velocity weight")
    assert "def reward(" in result.reward_definition
    assert result.proposed_changes == [
        "Scale forward speed more aggressively.",
        "Add a small upright posture bonus from observation[0].",
    ]


def test_openai_reward_designer_rejects_unsupported_parameters() -> None:
    """Model-backed generation should reject callable signatures the runner cannot satisfy."""

    class FakeOpenAIClient:
        """Fake model client returning an unsupported callable signature."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Return a JSON payload with an invalid reward parameter."""

            del request
            return ChatCompletionResponse(
                content=(
                    '{"reward_definition": '
                    '"def reward(observation, forbidden_signal):\\n    return 1.0\\n", '
                    '"change_summary": "Invalid change.", '
                    '"proposed_changes": ["Use a forbidden signal."]}'
                ),
                raw_response=None,
            )

    designer = OpenAIRewardDesigner(
        openai_client=FakeOpenAIClient(),
        config=RewardDesignConfig(mode=RewardDesignerMode.OPENAI),
    )

    with pytest.raises(RuntimeError, match="unsupported callable parameters"):
        designer.design_next_candidate(build_request())


def test_openai_reward_designer_requires_credentials() -> None:
    """OpenAI mode should fail clearly when no API key is available."""

    class NoCredentialClient:
        """Fake model client advertising no available credentials."""

        has_credentials = False

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Fail the test if a request is attempted without credentials."""

            del request
            raise AssertionError("chat_completion should not be called without credentials")

    designer = OpenAIRewardDesigner(
        openai_client=NoCredentialClient(),
        config=RewardDesignConfig(mode=RewardDesignerMode.OPENAI),
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        designer.design_next_candidate(build_request())


def test_openai_reward_designer_retries_after_invalid_model_output() -> None:
    """Designer should retry when the first model response fails signature validation."""

    responses = [
        ChatCompletionResponse(
            content="   ",
            raw_response=None,
        ),
        ChatCompletionResponse(
            content=(
                '{"reward_definition": '
                '"def reward(observation, x_velocity, reward_alive, reward_quadctrl, '
                'terminated):\\n    if terminated:\\n        return -5.0\\n    '
                'return float(x_velocity + reward_alive - reward_quadctrl)\\n", '
                '"change_summary": "Drop kwargs and use supported parameters.", '
                '"proposed_changes": ["Use explicit supported parameters."]}'
            ),
            raw_response=None,
        ),
    ]
    call_count = 0

    class FakeOpenAIClient:
        """Fake model client that returns one invalid response followed by a valid one."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Return a deterministic response sequence for retry coverage."""

            nonlocal call_count
            call_count += 1
            assert request.response_format == {"type": "json_object"}
            return responses[call_count - 1]

    designer = OpenAIRewardDesigner(
        openai_client=FakeOpenAIClient(),
        config=RewardDesignConfig(mode=RewardDesignerMode.OPENAI),
    )

    result = designer.design_next_candidate(build_request())

    assert call_count == 2
    assert "Drop kwargs" in result.change_summary
