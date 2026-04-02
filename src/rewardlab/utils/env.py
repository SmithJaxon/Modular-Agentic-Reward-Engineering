"""
Summary: Lightweight environment-file loading helpers for local RewardLab runtime config.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import os
from pathlib import Path


def load_runtime_environment(*, start_path: Path | None = None) -> dict[str, str]:
    """Return runtime configuration with process environment taking precedence."""

    values = _load_dotenv_values(start_path=start_path)
    for key, value in os.environ.items():
        values[key] = value
    return values


def _load_dotenv_values(*, start_path: Path | None = None) -> dict[str, str]:
    """Load key-value pairs from the configured `.env` file when present."""

    env_file = _find_dotenv_file(start_path=start_path)
    if env_file is None:
        return {}
    return _parse_dotenv_file(env_file)


def _find_dotenv_file(*, start_path: Path | None = None) -> Path | None:
    """Return the nearest `.env` file or an explicit override when configured."""

    explicit_path = os.getenv("REWARDLAB_ENV_FILE")
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return candidate
        return None

    current = (start_path or Path.cwd()).resolve()
    search_root = current.parent if current.is_file() else current
    for directory in (search_root, *search_root.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    """Parse a small `.env` file into string key-value pairs."""

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue

        key, raw_value = raw_line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        values[normalized_key] = _normalize_env_value(raw_value.strip())
    return values


def _normalize_env_value(raw_value: str) -> str:
    """Strip one layer of matching quotes from a parsed `.env` value."""

    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {'"', "'"}:
        return raw_value[1:-1]
    return raw_value
