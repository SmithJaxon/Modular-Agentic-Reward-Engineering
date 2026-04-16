"""
Summary: Live OpenAI runtime smoke validation gated behind an explicit opt-in environment flag.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os

import pytest

from rewardlab.llm.openai_client import OpenAIClient, OpenAIClientConfig
from rewardlab.utils.env_loader import load_project_dotenv


@pytest.mark.integration
def test_openai_runtime_smoke_via_project_client() -> None:
    """
    Verify the project OpenAI wrapper can complete a live reflection request
    when explicitly enabled.
    """
    if os.getenv("REWARDLAB_ENABLE_OPENAI_LIVE_TESTS", "").strip() != "1":
        pytest.skip("set REWARDLAB_ENABLE_OPENAI_LIVE_TESTS=1 to enable live OpenAI smoke tests")
    load_project_dotenv()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        pytest.skip("OPENAI_API_KEY is not set in the environment")
    pytest.importorskip("openai")

    client = OpenAIClient(
        OpenAIClientConfig(
            model=os.getenv("REWARDLAB_OPENAI_SMOKE_MODEL", "gpt-4o-mini").strip()
            or "gpt-4o-mini"
        )
    )
    response = client.generate_reflection(
        "Reply with exactly 'rewardlab-live-smoke-ok' and nothing else."
    )

    assert "rewardlab-live-smoke-ok" in response.lower()
