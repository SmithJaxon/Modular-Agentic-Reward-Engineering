"""
Summary: Unit tests for the OpenAI client wrapper error handling and response normalization.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import rewardlab.llm.openai_client as openai_client_module
from rewardlab.llm.openai_client import OpenAIClient, OpenAIClientConfig


class _FakeOpenAIError(Exception):
    """
    Represent a fake OpenAI SDK base error for wrapper tests.
    """


class _FakeAuthenticationError(_FakeOpenAIError):
    """
    Represent a fake authentication failure from the OpenAI SDK.
    """


def _install_fake_openai_module(response_factory: object) -> None:
    """
    Install a fake `openai` module into `sys.modules` for wrapper tests.
    """
    module = ModuleType("openai")

    class _FakeClient:
        """
        Minimal client object exposing the `responses.create` path used by the wrapper.
        """

        def __init__(self, api_key: str) -> None:
            """
            Store the provided API key for parity with the real client signature.
            """
            self.api_key = api_key
            self.responses = SimpleNamespace(create=response_factory)

    module.OpenAI = _FakeClient  # type: ignore[attr-defined]
    module.OpenAIError = _FakeOpenAIError  # type: ignore[attr-defined]
    sys.modules["openai"] = module


def test_openai_client_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify the wrapper raises a runtime error when the configured API key is absent.
    """
    monkeypatch.delenv("TEST_OPENAI_API_KEY", raising=False)

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    with pytest.raises(RuntimeError, match="missing required environment variable"):
        client._resolve_api_key()  # noqa: SLF001 - targeted unit validation.


def test_openai_client_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify the wrapper normalizes missing optional dependency failures.
    """
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(
        openai_client_module,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"{name} unavailable")),
    )

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    with pytest.raises(RuntimeError, match="openai package is not installed"):
        client.generate_reflection("hello")


def test_openai_client_loads_api_key_from_project_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify the wrapper consults the shared project `.env` loader before failing.
    """
    monkeypatch.delenv("TEST_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        openai_client_module,
        "load_project_dotenv",
        lambda: monkeypatch.setenv("TEST_OPENAI_API_KEY", "loaded-from-dotenv") or {},
    )

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    assert client._resolve_api_key() == "loaded-from-dotenv"  # noqa: SLF001


def test_openai_client_returns_trimmed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify the wrapper returns normalized response text from the SDK payload.
    """
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")

    def _create(**_: object) -> object:
        """
        Return a fake response object with leading and trailing whitespace.
        """
        return SimpleNamespace(output_text="  reflection ready  \n")

    _install_fake_openai_module(_create)
    monkeypatch.syspath_prepend(str(Path.cwd()))

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    assert client.generate_reflection("hello") == "reflection ready"


def test_openai_client_generate_text_with_usage_extracts_usage_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify usage counters are extracted from SDK response objects.
    """
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")

    def _create(**_: object) -> object:
        """
        Return a fake response object with usage-like fields.
        """
        usage = SimpleNamespace(input_tokens=12, output_tokens=7, total_cost=0.05)
        return SimpleNamespace(output_text="  done  ", usage=usage, model="gpt-5.4-mini")

    _install_fake_openai_module(_create)
    monkeypatch.syspath_prepend(str(Path.cwd()))

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))
    response = client.generate_text_with_usage("hello", max_output_tokens=20)
    assert response.text == "done"
    assert response.model_used == "gpt-5.4-mini"
    assert response.api_input_tokens == 12
    assert response.api_output_tokens == 7
    assert response.api_cost_usd == pytest.approx(0.05)


def test_openai_client_rejects_whitespace_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify the wrapper treats whitespace-only output as an empty API response.
    """
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")

    def _create(**_: object) -> object:
        """
        Return a fake response object containing only whitespace.
        """
        return SimpleNamespace(output_text="   \n\t")

    _install_fake_openai_module(_create)
    monkeypatch.syspath_prepend(str(Path.cwd()))

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    with pytest.raises(RuntimeError, match="empty reflection response"):
        client.generate_reflection("hello")


def test_openai_client_wraps_sdk_errors_without_echoing_raw_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify SDK errors are normalized to the wrapper runtime contract.
    """
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")

    def _create(**_: object) -> object:
        """
        Raise a fake authentication error containing sensitive-looking detail.
        """
        raise _FakeAuthenticationError("Incorrect API key provided: sk-secret-123")

    _install_fake_openai_module(_create)
    monkeypatch.syspath_prepend(str(Path.cwd()))

    client = OpenAIClient(config=OpenAIClientConfig(api_key_env="TEST_OPENAI_API_KEY"))

    with pytest.raises(RuntimeError, match="OpenAI request failed \\(_FakeAuthenticationError\\)"):
        client.generate_reflection("hello")
