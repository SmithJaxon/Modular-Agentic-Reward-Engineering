"""
Summary: OpenAI API wrapper for reflection generation with environment-key auth.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class OpenAIClientConfig:
    """
    Hold OpenAI client runtime configuration.
    """

    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


class OpenAIClient:
    """
    Provide a narrow wrapper around the OpenAI responses API.
    """

    def __init__(self, config: OpenAIClientConfig | None = None) -> None:
        """
        Initialize client wrapper and defer import until runtime use.

        Args:
            config: Optional OpenAI client runtime configuration.
        """
        self._config = config or OpenAIClientConfig()

    def _resolve_api_key(self) -> str:
        """
        Resolve API key from configured environment variable.

        Returns:
            API key string.

        Raises:
            RuntimeError: If key is missing.
        """
        value = os.getenv(self._config.api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"missing required environment variable: {self._config.api_key_env}")
        return value

    def generate_reflection(self, prompt: str) -> str:
        """
        Request concise reflection text from configured OpenAI model.

        Args:
            prompt: Reflection prompt text.

        Returns:
            Reflection text from the first output segment.

        Raises:
            RuntimeError: If the OpenAI package is unavailable or output is empty.
        """
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed; "
                "install optional dependency with 'pip install -e .[llm]'"
            ) from exc

        client = OpenAI(api_key=self._resolve_api_key())
        response: Any = client.responses.create(
            model=self._config.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=300,
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise RuntimeError("empty reflection response from OpenAI API")
        return text.strip()
