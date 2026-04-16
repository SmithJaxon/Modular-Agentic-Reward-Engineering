"""
Summary: Lightweight project .env loader for local validation and optional runtime configuration.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
from pathlib import Path


def default_dotenv_path() -> Path:
    """
    Resolve the repository-local `.env` file path.

    Returns:
        Absolute path to the project `.env` file.
    """
    return Path(__file__).resolve().parents[3] / ".env"


def load_project_dotenv(
    env_path: Path | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """
    Load simple `KEY=VALUE` pairs from the project `.env` file into `os.environ`.

    Args:
        env_path: Optional explicit path to an env file.
        override: Whether loaded values may replace existing environment values.

    Returns:
        Mapping of keys loaded into the current process environment.
    """
    path = env_path or default_dotenv_path()
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue

        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {"'", '"'}
        ):
            normalized_value = normalized_value[1:-1]

        if override or normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value
            loaded[normalized_key] = normalized_value

    return loaded
