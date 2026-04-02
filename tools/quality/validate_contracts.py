"""
Summary: Validate required contract artifacts and JSON schema parseability.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    """
    Load a JSON document from disk.

    Args:
        path: Absolute or relative path to a JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If JSON cannot be parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def validate_contracts(contracts_dir: Path) -> list[str]:
    """
    Validate required contract files and schema integrity.

    Args:
        contracts_dir: Directory containing generated contract artifacts.

    Returns:
        List of validation errors. Empty list indicates success.
    """
    errors: list[str] = []
    required_files = [
        contracts_dir / "orchestrator-cli.md",
        contracts_dir / "session-config.schema.json",
        contracts_dir / "session-report.schema.json",
    ]

    for path in required_files:
        if not path.exists():
            errors.append(f"Missing required contract file: {path}")

    for schema_path in required_files[1:]:
        if schema_path.exists():
            try:
                _load_json(schema_path)
            except ValueError as exc:
                errors.append(str(exc))

    return errors


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for contract validation utility.

    Returns:
        Configured argparse parser instance.
    """
    parser = argparse.ArgumentParser(description="Validate rewardlab contracts directory.")
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=Path("specs/001-iterative-reward-design/contracts"),
        help="Path to contract artifacts directory.",
    )
    return parser


def main() -> int:
    """
    Run validation and print results.

    Returns:
        Process exit code. Zero indicates success.
    """
    parser = _build_parser()
    args = parser.parse_args()
    errors = validate_contracts(args.contracts_dir)
    if errors:
        for err in errors:
            print(err)
        return 1
    print("Contracts validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
