"""
Summary: Environment-driven OpenAI client wrapper for RewardLab workflows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsChatCompletions(Protocol):
    """Protocol for the minimal OpenAI chat-completions surface used by RewardLab."""

    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion request."""


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
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionResponse:
    """Represents the normalized response returned by the wrapper."""

    content: str
    raw_response: Any


class OpenAIClient:
    """Test-friendly wrapper around the OpenAI SDK.

    The wrapper does not perform any network activity during construction.
    Callers must invoke a request method explicitly, which keeps development
    paths offline-safe and allows tests to inject a fake SDK client.
    """

    def __init__(
        self,
        config: OpenAIClientConfig | None = None,
        client: SupportsChatCompletions | Any | None = None,
    ) -> None:
        """Create a client wrapper from config or an injected SDK-compatible client."""

        self._config = config or self.config_from_environment()
        self._client = client

    @classmethod
    def config_from_environment(cls) -> OpenAIClientConfig:
        """Load runtime credentials from environment variables."""

        return OpenAIClientConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout_seconds=_parse_timeout(os.getenv("OPENAI_TIMEOUT_SECONDS")),
        )

    @property
    def config(self) -> OpenAIClientConfig:
        """Return the active runtime configuration."""

        return self._config

    @property
    def has_credentials(self) -> bool:
        """Return whether an API key is available for real requests."""

        return self._config.has_credentials

    def with_client(self, client: SupportsChatCompletions | Any) -> OpenAIClient:
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

        client = self._client or self.build_client()
        payload = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        response = client.chat.completions.create(**payload)
        content = _extract_message_content(response)
        return ChatCompletionResponse(content=content, raw_response=response)


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
