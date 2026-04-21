"""User-level startup shims for Python runtimes older than 3.11."""

from __future__ import annotations

import datetime as _datetime
import dataclasses as _dataclasses
import enum as _enum
import typing as _typing

try:
    import typing_extensions as _typing_extensions
except Exception:  # pragma: no cover - dependency guard
    _typing_extensions = None


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
