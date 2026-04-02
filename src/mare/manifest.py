from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class ExperimentManifest:
    """Metadata describing a single reproducible run."""

    name: str
    environment: str
    baseline: str
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")

