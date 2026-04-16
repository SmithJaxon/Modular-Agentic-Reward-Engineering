"""
Summary: Environment-driven OpenAI client wrapper for RewardLab workflows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rewardlab.utils.env import load_runtime_environment


class SupportsChatCompletions(Protocol):
    """Protocol for the minimal chat-completions create surface."""

    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion request."""


class SupportsChatNamespace(Protocol):
    """Protocol for the nested chat.completions namespace used by the client."""

    completions: SupportsChatCompletions


class SupportsChatClient(Protocol):
    """Protocol for the root client object used by the wrapper."""

    chat: SupportsChatNamespace


@dataclass(frozen=True, slots=True)
class OpenAIClientConfig:
    """Runtime configuration required to construct an OpenAI client."""

    api_key: str | None
    organization: str | None = None
    project: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = 30.0

    @property
    def has_credentials(self) -> bool:
        """Return whether the configuration includes the minimum credential set."""

        return bool(self.api_key)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Represents one message in a chat completion conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    """Represents a small, typed chat completion request."""

    model: str
    messages: tuple[ChatMessage, ...]
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionResponse:
    """Represents the normalized response returned by the wrapper."""

    content: str
    raw_response: Any
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class OpenAIClient:
    """Test-friendly wrapper around the OpenAI SDK.

    The wrapper does not perform any network activity during construction.
    Callers must invoke a request method explicitly, which keeps development
    paths offline-safe and allows tests to inject a fake SDK client.
    """

    def __init__(
        self,
        config: OpenAIClientConfig | None = None,
        client: SupportsChatClient | None = None,
    ) -> None:
        """Create a client wrapper from config or an injected SDK-compatible client."""

        self._config = config or self.config_from_environment()
        self._client = client

    @classmethod
    def config_from_environment(cls) -> OpenAIClientConfig:
        """Load runtime credentials from environment variables."""

        env = load_runtime_environment()
        return OpenAIClientConfig(
            api_key=env.get("OPENAI_API_KEY"),
            organization=env.get("OPENAI_ORG_ID") or env.get("OPENAI_ORGANIZATION"),
            project=env.get("OPENAI_PROJECT"),
            base_url=env.get("OPENAI_BASE_URL"),
            timeout_seconds=_parse_timeout(env.get("OPENAI_TIMEOUT_SECONDS")),
        )

    @property
    def config(self) -> OpenAIClientConfig:
        """Return the active runtime configuration."""

        return self._config

    @property
    def has_credentials(self) -> bool:
        """Return whether an API key is available for real requests."""

        return self._config.has_credentials

    def with_client(self, client: SupportsChatClient) -> OpenAIClient:
        """Return a copy of this wrapper that uses the provided SDK-compatible client."""

        return OpenAIClient(config=self._config, client=client)

    def build_client(self) -> Any:
        """Construct an OpenAI SDK client from the current configuration.

        This method intentionally does not attempt to contact the network.
        It only instantiates the SDK client when credentials are available.
        """

        if not self._config.api_key:
            raise RuntimeError("OPENAI_API_KEY is required before creating a live client")

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.organization:
            kwargs["organization"] = self._config.organization
        if self._config.project:
            kwargs["project"] = self._config.project
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        if self._config.timeout_seconds is not None:
            kwargs["timeout"] = self._config.timeout_seconds
        return OpenAI(**kwargs)

    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send one chat completion request and normalize the response."""

        client: Any = self._client or self.build_client()
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_completion_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        response = client.chat.completions.create(**payload)
        content = _extract_message_content(response)
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
        return ChatCompletionResponse(
            content=content,
            raw_response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def _parse_timeout(raw_value: str | None) -> float | None:
    """Parse an optional floating-point timeout value from the environment."""

    if raw_value is None or raw_value == "":
        return None
    try:
        return float(raw_value)
    except ValueError:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a numeric value") from None


def _extract_message_content(response: Any) -> str:
    """Extract the first assistant message content from a chat completion response."""

    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except Exception as exc:  # pragma: no cover - defensive around SDK shape changes
        raise ValueError("Unexpected OpenAI chat completion response shape") from exc

    if isinstance(content, list):
        return "".join(str(part) for part in content)
    if content is None:
        return ""
    return str(content)


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Extract normalized token usage counters from an SDK response when present."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return (
        int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        int(completion_tokens) if isinstance(completion_tokens, int) else None,
        int(total_tokens) if isinstance(total_tokens, int) else None,
    )
