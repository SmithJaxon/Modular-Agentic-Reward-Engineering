"""
Summary: Tests for the Python header and routine docstring audit utility.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "quality" / "check_headers.py"


@pytest.fixture()
def tool_module():
    """Load the audit utility module from its file path."""

    spec = importlib.util.spec_from_file_location("check_headers_tool", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str) -> Path:
    """Write text to a file and return the path for chaining."""

    path.write_text(content, encoding="utf-8")
    return path


def test_audit_reports_missing_module_header_and_routine_docstring(tmp_path: Path, tool_module) -> None:
    """The tool should flag missing module headers and missing docstrings."""

    target = write_file(
        tmp_path / "broken.py",
        """
def useful(value):
    total = value + 1
    return total
""".lstrip(),
    )

    issues = tool_module.audit_python_file(target)

    codes = {issue.code for issue in issues}
    assert "MISSING_MODULE_HEADER" in codes
    assert "MISSING_ROUTINE_DOCSTRING" in codes


def test_audit_accepts_valid_module_header_and_trivial_helpers(tmp_path: Path, tool_module) -> None:
    """The tool should allow valid headers and trivial routines without docstrings."""

    target = write_file(
        tmp_path / "good.py",
        '''
"""
Summary: Example utility module.
Created: 2026-04-02
Last Updated: 2026-04-02
"""


def helper():
    return 1


def documented(value):
    """Return the value after a small transformation."""

    if value:
        return value + 1
    return value
'''.lstrip(),
    )

    issues = tool_module.audit_python_file(target)

    assert issues == []


def test_collect_python_files_accepts_directories_and_files(tmp_path: Path, tool_module) -> None:
    """The path collector should expand directories and deduplicate results."""

    root = tmp_path / "pkg"
    root.mkdir()
    a = write_file(
        root / "a.py",
        '"""Summary: A.\nCreated: 2026-04-02\nLast Updated: 2026-04-02\n"""\n',
    )
    b = write_file(
        root / "nested.py",
        '"""Summary: B.\nCreated: 2026-04-02\nLast Updated: 2026-04-02\n"""\n',
    )

    files = tool_module.collect_python_files([root, a])

    assert files == [a.resolve(), b.resolve()]
