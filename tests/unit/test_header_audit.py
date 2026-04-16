"""
Summary: Regression tests for the Python module header and docstring audit utility.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_HEADERS_PATH = REPO_ROOT / "tools" / "quality" / "check_headers.py"


def _load_tool_module(name: str, path: Path) -> ModuleType:
    """
    Load a repository tool module from a concrete file path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_header_audit_accepts_well_formed_python_file() -> None:
    """
    Verify the audit utility accepts files with the required header and routine docs.
    """
    tool = _load_tool_module("check_headers_tool_valid", CHECK_HEADERS_PATH)
    root = REPO_ROOT / ".tmp-header-audit" / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    module_path = root / "valid_module.py"
    module_path.write_text(
        "\n".join(
            [
                '"""',
                "Summary: Example module for header audit validation.",
                "Created: 2026-04-02",
                "Last Updated: 2026-04-02",
                '"""',
                "",
                "class Example:",
                '    """Example public class."""',
                "",
                "    def public_method(self) -> int:",
                '        """Return a deterministic integer."""',
                "        return 1",
                "",
                "def public_function() -> str:",
                '    """Return deterministic text."""',
                '    return "ok"',
            ]
        ),
        encoding="utf-8",
    )

    assert tool.check_file(module_path) == []
    shutil.rmtree(root, ignore_errors=True)


def test_header_audit_reports_missing_module_header_and_function_docstring() -> None:
    """
    Verify the audit utility reports missing module and public routine documentation.
    """
    tool = _load_tool_module("check_headers_tool_invalid", CHECK_HEADERS_PATH)
    root = REPO_ROOT / ".tmp-header-audit" / uuid4().hex
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    module_path = root / "invalid_module.py"
    module_path.write_text(
        "\n".join(
            [
                "def public_function() -> str:",
                '    return "missing docs"',
            ]
        ),
        encoding="utf-8",
    )

    errors = tool.check_file(module_path)

    assert f"{module_path}: missing module docstring header" in errors
    assert any("missing docstring for function public_function" in error for error in errors)
    shutil.rmtree(root, ignore_errors=True)
