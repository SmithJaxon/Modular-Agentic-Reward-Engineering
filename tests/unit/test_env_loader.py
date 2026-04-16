"""
Summary: Unit tests for the lightweight project `.env` loader utility.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from rewardlab.utils.env_loader import load_project_dotenv


def test_load_project_dotenv_populates_missing_values() -> None:
    """
    Verify simple key/value pairs are loaded into the process environment.
    """
    root = Path(".tmp-env-loader") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    env_path = root / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "OPENAI_API_KEY = test-key",
                'REWARDLAB_OPENAI_SMOKE_MODEL="gpt-4o-mini"',
            ]
        ),
        encoding="utf-8",
    )

    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("REWARDLAB_OPENAI_SMOKE_MODEL", None)
    loaded = load_project_dotenv(env_path=env_path)

    assert loaded == {
        "OPENAI_API_KEY": "test-key",
        "REWARDLAB_OPENAI_SMOKE_MODEL": "gpt-4o-mini",
    }
    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert os.environ["REWARDLAB_OPENAI_SMOKE_MODEL"] == "gpt-4o-mini"
    shutil.rmtree(root, ignore_errors=True)


def test_load_project_dotenv_does_not_override_existing_values_by_default() -> None:
    """
    Verify existing environment variables win unless override is requested.
    """
    root = Path(".tmp-env-loader") / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    env_path = root / ".env"
    env_path.write_text("OPENAI_API_KEY=from-file", encoding="utf-8")

    os.environ["OPENAI_API_KEY"] = "from-env"
    loaded = load_project_dotenv(env_path=env_path)

    assert loaded == {}
    assert os.environ["OPENAI_API_KEY"] == "from-env"
    shutil.rmtree(root, ignore_errors=True)
