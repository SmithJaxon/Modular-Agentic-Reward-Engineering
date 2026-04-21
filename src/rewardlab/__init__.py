"""
Summary: Top-level package metadata for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import dataclasses as _dataclasses
import datetime as _datetime
import enum as _enum
import typing as _typing

try:
    import typing_extensions as _typing_extensions
except Exception:  # pragma: no cover - dependency guard
    _typing_extensions = None

__all__ = ["__version__"]

__version__ = "0.1.0"


def _apply_runtime_compatibility() -> None:
    """Backfill symbols/features missing on Python <3.11 runtimes."""

    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

    if not hasattr(_enum, "StrEnum"):
        class _CompatStrEnum(str, _enum.Enum):
            pass

        _enum.StrEnum = _CompatStrEnum  # type: ignore[attr-defined]

    if not hasattr(_typing, "Self") and _typing_extensions is not None:
        _typing.Self = _typing_extensions.Self  # type: ignore[attr-defined]

    if "slots" not in _dataclasses.dataclass.__code__.co_varnames:
        _orig_dataclass = _dataclasses.dataclass

        def _compat_dataclass(*args, **kwargs):
            kwargs.pop("slots", None)
            return _orig_dataclass(*args, **kwargs)

        _dataclasses.dataclass = _compat_dataclass  # type: ignore[assignment]


_apply_runtime_compatibility()
