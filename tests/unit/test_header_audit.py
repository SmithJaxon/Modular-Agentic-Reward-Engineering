"""
Summary: Regression tests for the repository-wide file header and docstring audit.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "quality" / "check_headers.py"


def load_audit_module():
    """Load the header audit tool directly from disk for regression testing."""

    spec = importlib.util.spec_from_file_location("check_headers_tool", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_header_audit_has_no_findings() -> None:
    """The header audit should report no issues across the maintained code paths."""

    module = load_audit_module()
    root = Path(__file__).resolve().parents[2]

    issues = module.audit_paths([root / "src", root / "tests", root / "tools"])

    assert issues == []
