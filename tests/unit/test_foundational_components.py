"""
Summary: Foundational unit tests for schemas, retry utilities, and state transitions.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rewardlab.feedback.peer_feedback_client import PeerFeedbackClient
from rewardlab.llm.openai_client import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    OpenAIClient,
    OpenAIClientConfig,
)
from rewardlab.orchestrator.state_machine import TransitionRequest, apply_transition, can_transition
from rewardlab.persistence.session_repository import RepositoryPaths, SessionRepository
from rewardlab.schemas.session_config import (
    EnvironmentBackend,
    FeedbackGate,
    SessionConfig,
    SessionRecord,
    SessionStatus,
    StopReason,
)
from rewardlab.schemas.session_report import (
    IterationSummary,
    ReportStatus,
    RiskLevel,
    SelectionCandidate,
    SessionReport,
)
from rewardlab.utils.retry import RetryError, RetryPolicy, compute_backoff_delays, retry_call


def build_session_record(**overrides: object) -> SessionRecord:
    """Create a valid persisted session record for focused tests."""

    payload: dict[str, object] = {
        "session_id": "session-001",
        "objective_text": "Improve the CartPole reward.",
        "environment_id": "cartpole-v1",
        "environment_backend": EnvironmentBackend.GYMNASIUM,
        "no_improve_limit": 3,
        "max_iterations": 8,
        "feedback_gate": FeedbackGate.NONE,
        "status": SessionStatus.DRAFT,
        "metadata": {"seed": 7},
    }
    payload.update(overrides)
    return SessionRecord.model_validate(payload)


def test_session_config_rejects_blank_objective_text() -> None:
    """Session configuration should reject blank text after stripping."""

    with pytest.raises(ValidationError):
        SessionConfig(
            objective_text="   ",
            environment_id="cartpole-v1",
            environment_backend=EnvironmentBackend.GYMNASIUM,
            no_improve_limit=3,
            max_iterations=10,
            feedback_gate=FeedbackGate.NONE,
        )


def test_session_record_requires_stop_reason_for_paused_status() -> None:
    """Paused sessions should require an explicit stop reason."""

    with pytest.raises(ValidationError):
        build_session_record(status=SessionStatus.PAUSED)


def test_session_report_rejects_duplicate_iteration_indices() -> None:
    """Session reports should require unique iteration indices."""

    with pytest.raises(ValidationError):
        SessionReport(
            session_id="session-001",
            status=ReportStatus.COMPLETED,
            stop_reason=StopReason.CONVERGENCE,
            environment_backend=EnvironmentBackend.GYMNASIUM,
            best_candidate=SelectionCandidate(
                candidate_id="candidate-001",
                aggregate_score=1.5,
                selection_summary="Selected after stable improvement.",
            ),
            iterations=[
                IterationSummary(
                    iteration_index=0,
                    candidate_id="candidate-000",
                    performance_summary="Baseline run.",
                    risk_level=RiskLevel.LOW,
                ),
                IterationSummary(
                    iteration_index=0,
                    candidate_id="candidate-001",
                    performance_summary="Improved run.",
                    risk_level=RiskLevel.LOW,
                ),
            ],
        )


def test_compute_backoff_delays_grows_until_capped() -> None:
    """Backoff delays should grow geometrically and then cap."""

    policy = RetryPolicy(max_attempts=4, base_delay_seconds=0.5, max_delay_seconds=1.0)

    assert compute_backoff_delays(policy) == [0.5, 1.0, 1.0]


def test_retry_call_retries_then_returns_result() -> None:
    """Retry logic should rerun transient failures until the operation succeeds."""

    attempts: list[int] = []
    sleeps: list[float] = []

    def operation() -> str:
        """Fail twice before returning a successful result."""

        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    result = retry_call(
        operation,
        policy=RetryPolicy(max_attempts=4, base_delay_seconds=0.25),
        retryable_exceptions=(RuntimeError,),
        sleeper=sleeps.append,
    )

    assert result == "ok"
    assert attempts == [1, 2, 3]
    assert sleeps == [0.25, 0.5]


def test_retry_call_raises_retry_error_after_exhaustion() -> None:
    """Retry logic should expose captured failures after the final attempt."""

    def operation() -> None:
        """Always fail so retry exhaustion can be asserted."""

        raise RuntimeError("still failing")

    with pytest.raises(RetryError) as exc_info:
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.1),
            retryable_exceptions=(RuntimeError,),
            sleeper=lambda _: None,
        )

    assert len(exc_info.value.failures) == 2
    assert exc_info.value.failures[-1].attempt_number == 2


def test_can_transition_allows_defined_session_lifecycle_edges() -> None:
    """The state machine should accept only declared lifecycle transitions."""

    assert can_transition(SessionStatus.DRAFT, SessionStatus.RUNNING) is True
    assert can_transition(SessionStatus.RUNNING, SessionStatus.PAUSED) is True
    assert can_transition(SessionStatus.PAUSED, SessionStatus.RUNNING) is True
    assert can_transition(SessionStatus.COMPLETED, SessionStatus.RUNNING) is False


def test_apply_transition_sets_runtime_fields_for_start_and_interrupt() -> None:
    """Transitions should stamp the expected lifecycle fields."""

    record = build_session_record()
    started_at = datetime(2026, 4, 2, 22, 0, tzinfo=UTC)
    running = apply_transition(
        record,
        TransitionRequest(next_status=SessionStatus.RUNNING, occurred_at=started_at),
    )

    assert running.status == SessionStatus.RUNNING
    assert running.started_at == started_at
    assert running.stop_reason is None

    ended_at = datetime(2026, 4, 2, 23, 0, tzinfo=UTC)
    interrupted = apply_transition(
        running,
        TransitionRequest(
            next_status=SessionStatus.INTERRUPTED,
            occurred_at=ended_at,
            stop_reason=StopReason.USER_INTERRUPT,
            best_candidate_id="candidate-007",
        ),
    )

    assert interrupted.status == SessionStatus.INTERRUPTED
    assert interrupted.ended_at == ended_at
    assert interrupted.best_candidate_id == "candidate-007"


def test_apply_transition_rejects_invalid_edge() -> None:
    """The state machine should reject undefined lifecycle transitions."""

    record = build_session_record(
        status=SessionStatus.COMPLETED,
        ended_at=datetime.now(UTC),
        stop_reason=StopReason.CONVERGENCE,
    )

    with pytest.raises(ValueError):
        apply_transition(record, TransitionRequest(next_status=SessionStatus.RUNNING))


def test_session_repository_round_trips_session_and_events(tmp_path) -> None:
    """The session repository should persist sessions and append/read events."""

    repository = SessionRepository(
        RepositoryPaths(
            database_path=tmp_path / "rewardlab.sqlite3",
            event_log_path=tmp_path / "events" / "session.jsonl",
        )
    )
    repository.initialize()

    started_at = datetime(2026, 4, 2, 22, 0, tzinfo=UTC)
    record = build_session_record(
        status=SessionStatus.RUNNING,
        started_at=started_at,
    )
    repository.save_session(record)
    repository.append_event(
        session_id=record.session_id,
        event_type="session.started",
        payload={"iteration": 0},
    )

    loaded = repository.get_session(record.session_id)
    events = repository.read_events(record.session_id)

    assert loaded is not None
    assert loaded.session_id == record.session_id
    assert loaded.status == SessionStatus.RUNNING
    assert len(events) == 1
    assert events[0].event_type == "session.started"


def test_openai_client_uses_injected_client_without_network() -> None:
    """The OpenAI wrapper should support fake injected clients for local tests."""

    captured_payload: dict[str, object] = {}

    class FakeCompletions:
        """Minimal fake completions surface for the wrapper test."""

        def create(self, **kwargs: object) -> object:
            """Return a response object with the shape expected by the wrapper."""

            captured_payload.update(kwargs)
            message = type("Message", (), {"content": "ok"})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        """Minimal fake root client exposing chat completions."""

        def __init__(self) -> None:
            """Provide the nested chat.completions namespace expected by the wrapper."""

            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = OpenAIClient(
        config=OpenAIClientConfig(api_key=None),
        client=FakeClient(),
    )

    response = client.chat_completion(
        ChatCompletionRequest(
            model="gpt-5-nano",
            messages=(ChatMessage(role="user", content="ping"),),
            reasoning_effort="minimal",
            max_tokens=12,
        )
    )

    assert isinstance(response, ChatCompletionResponse)
    assert response.content == "ok"
    assert captured_payload["reasoning_effort"] == "minimal"
    assert captured_payload["max_completion_tokens"] == 12
    assert "max_tokens" not in captured_payload


def test_peer_feedback_client_uses_gpt5_nano_without_temperature_override() -> None:
    """The live peer-review request should remain compatible with GPT-5 nano."""

    captured_request: ChatCompletionRequest | None = None

    class FakeOpenAIClient:
        """Minimal fake client for peer-feedback request inspection."""

        has_credentials = True

        def chat_completion(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            """Capture the outgoing request and return a stable response."""

            nonlocal captured_request
            captured_request = request
            return ChatCompletionResponse(content="Concise live critique.", raw_response=None)

    feedback = PeerFeedbackClient(FakeOpenAIClient()).request_feedback(
        session_id="session-001",
        candidate_id="candidate-001",
        objective_text="Reward stable, centered balance.",
        reward_definition="def reward(state): return 1.0",
        aggregate_score=1.2,
    )

    assert captured_request is not None
    assert captured_request.model == "gpt-5-nano"
    assert captured_request.reasoning_effort == "minimal"
    assert captured_request.max_tokens == 120
    assert captured_request.temperature is None
    assert feedback.comment == "Concise live critique."
    assert feedback.score == 0.85
