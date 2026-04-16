"""
Summary: Tests for the contract validation utility.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_validator_module():
    """Load the validator module directly from its file path."""

    module_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "quality"
        / "validate_contracts.py"
    )
    spec = importlib.util.spec_from_file_location("validate_contracts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_contract_files(contract_dir: Path) -> None:
    """Write a minimal valid contract set to the provided directory."""

    contract_dir.mkdir(parents=True, exist_ok=True)
    (contract_dir / "orchestrator-cli.md").write_text(
        "# Contract: Orchestrator CLI\n",
        encoding="utf-8",
    )
    (contract_dir / "session-config.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://rewardlab.local/contracts/session-config.schema.json",
                "title": "SessionConfig",
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "objective_text",
                    "environment_id",
                    "environment_backend",
                    "no_improve_limit",
                    "max_iterations",
                    "feedback_gate",
                ],
                "properties": {
                    "objective_text": {"type": "string", "minLength": 1},
                    "environment_id": {"type": "string", "minLength": 1},
                    "environment_backend": {
                        "type": "string",
                        "enum": ["gymnasium"],
                    },
                    "no_improve_limit": {"type": "integer", "minimum": 1},
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "feedback_gate": {
                        "type": "string",
                        "enum": ["none", "one_required", "both_required"],
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": {
                            "type": ["string", "number", "boolean"]
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (contract_dir / "session-report.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://rewardlab.local/contracts/session-report.schema.json",
                "title": "SessionReport",
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "session_id",
                    "status",
                    "stop_reason",
                    "environment_backend",
                    "best_candidate",
                    "iterations",
                ],
                "properties": {
                    "session_id": {"type": "string", "minLength": 1},
                    "status": {
                        "type": "string",
                        "enum": ["paused", "interrupted", "completed", "failed"],
                    },
                    "stop_reason": {
                        "type": "string",
                        "enum": [
                            "user_interrupt",
                            "convergence",
                            "iteration_cap",
                            "api_failure_pause",
                            "error",
                        ],
                    },
                    "environment_backend": {
                        "type": "string",
                        "enum": ["gymnasium"],
                    },
                    "best_candidate": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "candidate_id",
                            "aggregate_score",
                            "selection_summary",
                        ],
                        "properties": {
                            "candidate_id": {"type": "string", "minLength": 1},
                            "aggregate_score": {"type": "number"},
                            "selection_summary": {"type": "string", "minLength": 1},
                            "minor_robustness_risk_accepted": {"type": "boolean"},
                        },
                    },
                    "iterations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "iteration_index",
                                "candidate_id",
                                "performance_summary",
                                "risk_level",
                            ],
                            "properties": {
                                "iteration_index": {"type": "integer", "minimum": 0},
                                "candidate_id": {"type": "string", "minLength": 1},
                                "performance_summary": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "risk_level": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                                "feedback_count": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_validate_contract_directory_accepts_expected_schema(tmp_path):
    """The validator should accept the expected contract set."""

    module = load_validator_module()
    contract_dir = tmp_path / "specs" / "001-iterative-reward-design" / "contracts"
    write_contract_files(contract_dir)

    issues = module.validate_contract_directory(contract_dir)

    assert issues == []


def test_validate_contract_directory_reports_schema_shape_errors(tmp_path):
    """The validator should report JSON and schema-shape problems."""

    module = load_validator_module()
    contract_dir = tmp_path / "specs" / "001-iterative-reward-design" / "contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    (contract_dir / "orchestrator-cli.md").write_text(
        "# Contract: Orchestrator CLI\n",
        encoding="utf-8",
    )
    (contract_dir / "session-config.schema.json").write_text(
        '{"title": "SessionConfig", "type": "object", "additionalProperties": false}',
        encoding="utf-8",
    )
    (contract_dir / "session-report.schema.json").write_text(
        "not-json",
        encoding="utf-8",
    )

    issues = module.validate_contract_directory(contract_dir)
    messages = sorted(f"{issue.path.name}:{issue.message}" for issue in issues)

    assert any("session-config.schema.json" in message for message in messages)
    assert any("session-report.schema.json:invalid JSON" in message for message in messages)
    assert any("required fields do not match" in message for message in messages)
