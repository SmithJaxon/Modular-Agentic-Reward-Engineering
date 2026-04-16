"""
Summary: OpenAI API wrapper for reflection generation with environment-key auth.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from rewardlab.utils.env_loader import load_project_dotenv


@dataclass(slots=True, frozen=True)
class OpenAIClientConfig:
    """
    Hold OpenAI client runtime configuration.
    """

    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(slots=True, frozen=True)
class OpenAITextResponse:
    """
    Normalize text and usage fields returned by one OpenAI responses call.
    """

    text: str
    model_used: str | None
    api_input_tokens: int
    api_output_tokens: int
    api_cost_usd: float


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
        load_project_dotenv()
        value = os.getenv(self._config.api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"missing required environment variable: {self._config.api_key_env}")
        return value

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Request text from the configured OpenAI model.

        Args:
            prompt: Prompt text.
            max_output_tokens: Maximum response length budget.
            reasoning_effort: Optional reasoning-effort hint.

        Returns:
            Reflection text from the first output segment.

        Raises:
            RuntimeError: If the OpenAI package is unavailable, the request
                fails, or output is empty.
        """
        response = self.generate_text_with_usage(
            prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        return response.text

    def generate_text_with_usage(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> OpenAITextResponse:
        """
        Request text and normalized usage metrics from the OpenAI responses API.
        """
        try:
            openai_module = import_module("openai")
            OpenAI = openai_module.OpenAI
            OpenAIError = openai_module.OpenAIError
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed; "
                "install optional dependency with 'pip install -e .[llm]'"
            ) from exc

        client = OpenAI(api_key=self._resolve_api_key())
        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort is not None and reasoning_effort.strip():
            request_payload["reasoning"] = {"effort": reasoning_effort.strip()}
        try:
            try:
                response = client.responses.create(**request_payload)
            except TypeError:
                # Fallback for client versions that do not yet support `reasoning`.
                request_payload.pop("reasoning", None)
                response = client.responses.create(**request_payload)
        except OpenAIError as exc:
            raise RuntimeError(
                f"OpenAI request failed ({exc.__class__.__name__})"
            ) from exc
        try:
            text = getattr(response, "output_text", "").strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"OpenAI response parsing failed ({exc.__class__.__name__})"
            ) from exc
        if not text:
            raise RuntimeError("empty reflection response from OpenAI API")

        usage = getattr(response, "usage", None)
        input_tokens = _as_non_negative_int(_extract_usage_field(usage, "input_tokens"))
        output_tokens = _as_non_negative_int(_extract_usage_field(usage, "output_tokens"))
        cost_usd = _as_non_negative_float(_extract_usage_field(usage, "total_cost"))
        model_used_raw = getattr(response, "model", self._config.model)
        model_used = model_used_raw if isinstance(model_used_raw, str) else self._config.model
        return OpenAITextResponse(
            text=text,
            model_used=model_used,
            api_input_tokens=input_tokens,
            api_output_tokens=output_tokens,
            api_cost_usd=cost_usd,
        )

    def generate_reflection(self, prompt: str) -> str:
        """
        Request concise reflection text from configured OpenAI model.

        Args:
            prompt: Reflection prompt text.

        Returns:
            Reflection text from the first output segment.
        """
        return self.generate_text(prompt, max_output_tokens=300)


def _extract_usage_field(usage: object, key: str) -> object:
    """
    Extract one usage field from object- or dict-shaped SDK payloads.
    """
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(key, 0)
    return getattr(usage, key, 0)


def _as_non_negative_int(raw: object) -> int:
    """
    Coerce numeric usage fields into non-negative integers.
    """
    if not isinstance(raw, int | float):
        return 0
    return max(0, int(raw))


def _as_non_negative_float(raw: object) -> float:
    """
    Coerce numeric usage fields into non-negative floats.
    """
    if not isinstance(raw, int | float):
        return 0.0
    return max(0.0, float(raw))
