from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical locations used by the research prototype."""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def src(self) -> Path:
        return self.root / "src"

