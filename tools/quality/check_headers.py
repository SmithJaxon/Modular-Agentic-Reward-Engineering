"""
Summary: Validate Python file headers and routine docstrings for constitution compliance.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path

HEADER_KEYS = ("Summary:", "Created:", "Last Updated:")


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """
    Yield Python files under the provided targets.

    Args:
        paths: Candidate file or directory paths.

    Yields:
        Individual Python source files.
    """
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if path.is_dir():
            yield from path.rglob("*.py")


def _check_module_header(tree: ast.Module, file_path: Path) -> list[str]:
    """
    Validate required module header fields.

    Args:
        tree: Parsed AST for a module.
        file_path: Source file path for reporting.

    Returns:
        List of header-related errors.
    """
    errors: list[str] = []
    docstring = ast.get_docstring(tree, clean=False)
    if not docstring:
        return [f"{file_path}: missing module docstring header"]
    for key in HEADER_KEYS:
        if key not in docstring:
            errors.append(f"{file_path}: module header missing '{key}'")
    return errors


def _has_docstring(node: ast.AST) -> bool:
    """
    Determine whether an AST node contains a docstring.

    Args:
        node: Function, async function, or class definition node.

    Returns:
        True when a non-empty docstring exists.
    """
    return bool(ast.get_docstring(node, clean=False))


def _check_routine_docstrings(tree: ast.Module, file_path: Path) -> list[str]:
    """
    Validate function and method docstrings for non-private routines.

    Args:
        tree: Parsed AST for a module.
        file_path: Source file path for reporting.

    Returns:
        List of docstring-related errors.
    """
    errors: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_") and not _has_docstring(node):
                errors.append(
                    f"{file_path}:{node.lineno}: "
                    f"missing docstring for function {node.name}"
                )
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_") and not _has_docstring(node):
                errors.append(f"{file_path}:{node.lineno}: missing docstring for class {node.name}")
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    if child.name.startswith("_"):
                        continue
                    if not _has_docstring(child):
                        errors.append(
                            f"{file_path}:{child.lineno}: "
                            f"missing docstring for method {node.name}.{child.name}"
                        )
    return errors


def check_file(path: Path) -> list[str]:
    """
    Run all header and docstring checks for one file.

    Args:
        path: Python file to inspect.

    Returns:
        List of validation errors for the file.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"{path}: parse failure: {exc}"]

    errors = _check_module_header(tree, path)
    errors.extend(_check_routine_docstrings(tree, path))
    return errors


def run(paths: list[Path]) -> list[str]:
    """
    Execute checks against all requested files.

    Args:
        paths: Input files and/or directories.

    Returns:
        Flat list of validation errors.
    """
    failures: list[str] = []
    for file_path in _iter_python_files(paths):
        failures.extend(check_file(file_path))
    return failures


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for header checks.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Check Python file headers and docstrings.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to inspect.")
    return parser


def main() -> int:
    """
    Run CLI entrypoint and return process status.

    Returns:
        Exit status code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    failures = run(args.paths)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("Header and docstring checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
