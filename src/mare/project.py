from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AppConfig, load_config
from .paths import ProjectPaths


@dataclass(frozen=True)
class ProjectContext:
    """Convenience bundle for config and filesystem layout."""

    config: AppConfig
    paths: ProjectPaths


def load_project_context(project_root: Optional[Path] = None) -> ProjectContext:
    config = load_config(project_root=project_root)
    return ProjectContext(config=config, paths=ProjectPaths(config.project_root))
