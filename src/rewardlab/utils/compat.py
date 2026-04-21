"""Compatibility helpers shared across Python runtime versions."""

from __future__ import annotations

import enum

try:
    from rewardlab.utils.compat import StrEnum as StrEnum  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - py<3.11 fallback
    class StrEnum(str, enum.Enum):
        """Python <3.11 fallback for enum.StrEnum."""

        pass

