"""
Summary: Validate feature contract and schema artifacts.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "001-iterative-reward-design"
    / "contracts"
)

EXPECTED_FILES = {
    "session-config.schema.json",
    "session-report.schema.json",
    "orchestrator-cli.md",
}


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation problem."""

    path: Path
    message: str


def load_json_file(path: Path) -> Any:
    """Load a JSON document from disk."""

    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def validate_session_config_schema(document: Any) -> list[str]:
    """Validate the expected top-level structure for the session config schema."""

    errors: list[str] = []
    if not isinstance(document, dict):
        return ["root document must be a JSON object"]

    if document.get("title") != "SessionConfig":
        errors.append("title must be 'SessionConfig'")
    if document.get("type") != "object":
        errors.append("type must be 'object'")
    if document.get("additionalProperties") is not False:
        errors.append("additionalProperties must be false")

    required = document.get("required")
    expected_required = [
        "objective_text",
        "environment_id",
        "environment_backend",
        "no_improve_limit",
        "max_iterations",
        "feedback_gate",
    ]
    if required != expected_required:
        errors.append("required fields do not match the expected session config set")

    properties = document.get("properties")
    if not isinstance(properties, dict):
        errors.append("properties must be a JSON object")
        return errors

    for key in expected_required + ["metadata"]:
        if key not in properties:
            errors.append(f"missing property definition for {key}")

    expected_enums = {
        "environment_backend": ["gymnasium", "isaacgym"],
        "feedback_gate": ["none", "one_required", "both_required"],
    }
    for key, expected in expected_enums.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            errors.append(f"{key} must be defined as an object")
            continue
        if prop.get("enum") != expected:
            errors.append(f"{key} enum does not match expected values")

    return errors


def validate_session_report_schema(document: Any) -> list[str]:
    """Validate the expected top-level structure for the session report schema."""

    errors: list[str] = []
    if not isinstance(document, dict):
        return ["root document must be a JSON object"]

    if document.get("title") != "SessionReport":
        errors.append("title must be 'SessionReport'")
    if document.get("type") != "object":
        errors.append("type must be 'object'")
    if document.get("additionalProperties") is not False:
        errors.append("additionalProperties must be false")

    required = document.get("required")
    expected_required = [
        "session_id",
        "status",
        "stop_reason",
        "environment_backend",
        "best_candidate",
        "iterations",
    ]
    if required != expected_required:
        errors.append("required fields do not match the expected session report set")

    properties = document.get("properties")
    if not isinstance(properties, dict):
        errors.append("properties must be a JSON object")
        return errors

    for key in expected_required:
        if key not in properties:
            errors.append(f"missing property definition for {key}")

    status = properties.get("status")
    if not isinstance(status, dict) or status.get("enum") != [
        "paused",
        "interrupted",
        "completed",
        "failed",
    ]:
        errors.append("status enum does not match expected values")

    stop_reason = properties.get("stop_reason")
    if not isinstance(stop_reason, dict) or stop_reason.get("enum") != [
        "user_interrupt",
        "convergence",
        "iteration_cap",
        "api_failure_pause",
        "error",
    ]:
        errors.append("stop_reason enum does not match expected values")

    environment_backend = properties.get("environment_backend")
    if not isinstance(environment_backend, dict) or environment_backend.get("enum") != [
        "gymnasium",
        "isaacgym",
    ]:
        errors.append("environment_backend enum does not match expected values")

    best_candidate = properties.get("best_candidate")
    if not isinstance(best_candidate, dict):
        errors.append("best_candidate must be defined as an object")
    else:
        candidate_required = [
            "candidate_id",
            "aggregate_score",
            "selection_summary",
        ]
        if best_candidate.get("required") != candidate_required:
            errors.append("best_candidate required fields do not match expected values")

    iterations = properties.get("iterations")
    if not isinstance(iterations, dict):
        errors.append("iterations must be defined as an object")
    else:
        item_schema = iterations.get("items")
        if not isinstance(item_schema, dict):
            errors.append("iterations.items must be defined as an object")
        else:
            item_required = [
                "iteration_index",
                "candidate_id",
                "performance_summary",
                "risk_level",
            ]
            if item_schema.get("required") != item_required:
                errors.append("iteration item required fields do not match expected values")

    return errors


def _optional_jsonschema_validate(path: Path, document: Any) -> list[str]:
    """Optionally validate a schema document with jsonschema when installed."""

    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except Exception:
        return []

    try:
        Draft202012Validator.check_schema(document)
    except Exception as exc:  # pragma: no cover - exercised only when installed
        return [f"jsonschema validation failed for {path.name}: {exc}"]
    return []


def validate_contract_directory(contract_dir: Path | None = None) -> list[ValidationIssue]:
    """Validate the expected contract artifacts in the feature contract directory."""

    directory = contract_dir or CONTRACT_DIR
    issues: list[ValidationIssue] = []

    if not directory.exists():
        return [ValidationIssue(directory, "contract directory does not exist")]

    present_files = {item.name for item in directory.iterdir() if item.is_file()}
    missing = EXPECTED_FILES - present_files
    for filename in sorted(missing):
        issues.append(ValidationIssue(directory / filename, "file is missing"))

    session_config_path = directory / "session-config.schema.json"
    if session_config_path.exists():
        try:
            document = load_json_file(session_config_path)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(session_config_path, f"invalid JSON: {exc.msg}"))
        else:
            for message in validate_session_config_schema(document):
                issues.append(ValidationIssue(session_config_path, message))
            for message in _optional_jsonschema_validate(session_config_path, document):
                issues.append(ValidationIssue(session_config_path, message))

    session_report_path = directory / "session-report.schema.json"
    if session_report_path.exists():
        try:
            document = load_json_file(session_report_path)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(session_report_path, f"invalid JSON: {exc.msg}"))
        else:
            for message in validate_session_report_schema(document):
                issues.append(ValidationIssue(session_report_path, message))
            for message in _optional_jsonschema_validate(session_report_path, document):
                issues.append(ValidationIssue(session_report_path, message))

    return issues


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Validate contract and schema artifacts.")
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=CONTRACT_DIR,
        help="Path to the feature contract directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the contract validator and return an exit status."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    issues = validate_contract_directory(args.contract_dir)
    if issues:
        for issue in issues:
            print(f"{issue.path}: {issue.message}")
        return 1

    print(f"Validated {args.contract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
