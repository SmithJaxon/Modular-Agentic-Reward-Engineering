from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration loaded from environment variables."""

    project_root: Path
    openai_api_key: Optional[str]
    openai_model: str
    python_version_target: str
    isaacgym_home: Optional[Path]


def load_config(project_root: Optional[Path] = None) -> AppConfig:
    root = project_root or Path(__file__).resolve().parents[2]
    isaacgym_home = os.getenv("ISAACGYM_HOME")
    return AppConfig(
        project_root=root,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"),
        python_version_target=os.getenv("PYTHON_VERSION_TARGET", "3.11"),
        isaacgym_home=Path(isaacgym_home).expanduser() if isaacgym_home else None,
    )

