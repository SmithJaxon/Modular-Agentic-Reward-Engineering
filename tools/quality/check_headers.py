"""
Summary: Audit Python files for required module headers and non-trivial routine docstrings.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

import argparse
import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MODULE_HEADER_FIELDS = ("Summary:", "Created:", "Last Updated:")


@dataclass(frozen=True)
class Issue:
    """Represents a single header or docstring audit finding."""

    path: Path
    line: int
    code: str
    message: str


def collect_python_files(paths: Sequence[Path]) -> list[Path]:
    """Expand file and directory inputs into a sorted list of Python files."""

    collected: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_dir():
            candidates = sorted(
                candidate for candidate in path.rglob("*.py") if candidate.is_file()
            )
        elif path.suffix == ".py" and path.is_file():
            candidates = [path]
        else:
            candidates = []

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(resolved)

    return sorted(collected)


def audit_module_header(path: Path, source: str) -> list[Issue]:
    """Check that a module docstring contains the required file header fields."""

    issues: list[Issue] = []

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            Issue(
                path=path,
                line=exc.lineno or 1,
                code="SYNTAX_ERROR",
                message=f"Python syntax error: {exc.msg}",
            )
        ]

    docstring = ast.get_docstring(module, clean=False)
    if not docstring:
        return [
            Issue(
                path=path,
                line=1,
                code="MISSING_MODULE_HEADER",
                message=(
                    "Missing module docstring header with Summary, Created, "
                    "and Last Updated fields."
                ),
            )
        ]

    for field in MODULE_HEADER_FIELDS:
        if field not in docstring:
            issues.append(
                Issue(
                    path=path,
                    line=1,
                    code="INCOMPLETE_MODULE_HEADER",
                    message=f"Module docstring is missing required field: {field[:-1]}",
                )
            )

    created_line = _find_header_date(docstring, "Created")
    updated_line = _find_header_date(docstring, "Last Updated")
    if "Created:" in docstring and created_line is None:
        issues.append(
            Issue(
                path=path,
                line=1,
                code="INVALID_MODULE_HEADER",
                message="Created field must use ISO date format YYYY-MM-DD.",
            )
        )
    if "Last Updated:" in docstring and updated_line is None:
        issues.append(
            Issue(
                path=path,
                line=1,
                code="INVALID_MODULE_HEADER",
                message="Last Updated field must use ISO date format YYYY-MM-DD.",
            )
        )

    return issues


def audit_routine_docstrings(path: Path, source: str) -> list[Issue]:
    """Flag non-trivial functions and methods that are missing docstrings."""

    issues: list[Issue] = []
    module = ast.parse(source, filename=str(path))

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _needs_docstring(node):
            if ast.get_docstring(node, clean=False):
                continue

            routine_name = node.name
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    code="MISSING_ROUTINE_DOCSTRING",
                    message=f"Non-trivial routine '{routine_name}' is missing a docstring.",
                )
            )

    return issues


def audit_python_file(path: Path) -> list[Issue]:
    """Run all header and docstring checks for a single Python file."""

    source = path.read_text(encoding="utf-8")
    issues = audit_module_header(path, source)

    try:
        issues.extend(audit_routine_docstrings(path, source))
    except SyntaxError as exc:
        issues.append(
            Issue(
                path=path,
                line=exc.lineno or 1,
                code="SYNTAX_ERROR",
                message=f"Python syntax error: {exc.msg}",
            )
        )

    return issues


def audit_paths(paths: Sequence[Path]) -> list[Issue]:
    """Audit a collection of files or directories and return all findings."""

    issues: list[Issue] = []
    for path in collect_python_files(paths):
        issues.extend(audit_python_file(path))
    return issues


def format_issue(issue: Issue) -> str:
    """Format an audit finding for terminal output."""

    return f"{issue.path}:{issue.line}: {issue.code}: {issue.message}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the audit utility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python files or directories to audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit utility and return a process exit code."""

    args = parse_args(argv)
    issues = audit_paths(args.paths)
    for issue in issues:
        print(format_issue(issue))
    return 1 if issues else 0


def _find_header_date(docstring: str, label: str) -> str | None:
    """Return the ISO-formatted date line for a module header field if present."""

    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*$")
    for line in docstring.splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _needs_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Decide whether a routine is non-trivial enough to require a docstring."""

    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    meaningful = [statement for statement in body if not _is_trivial_noop(statement)]
    if not meaningful:
        return False
    if len(meaningful) > 1:
        return True
    return not isinstance(meaningful[0], (ast.Return, ast.Raise))


def _is_trivial_noop(statement: ast.stmt) -> bool:
    """Return True for statements that do not make a routine meaningfully non-trivial."""

    if isinstance(statement, ast.Pass):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


if __name__ == "__main__":
    raise SystemExit(main())
