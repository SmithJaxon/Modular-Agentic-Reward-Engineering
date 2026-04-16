"""
Summary: Regression tests that keep JSON contract artifacts aligned with runtime schemas.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.schemas.session_report import (
    BestCandidateReport,
    IterationReport,
    RiskLevel,
    SessionReport,
    SessionStatus,
    StopReason,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "specs" / "001-iterative-reward-design" / "contracts"
VALIDATE_CONTRACTS_PATH = REPO_ROOT / "tools" / "quality" / "validate_contracts.py"


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


def _load_json(path: Path) -> dict[str, object]:
    """
    Load a JSON artifact from disk.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_contract_validation_tool_accepts_feature_contract_directory() -> None:
    """
    Verify the contract validation utility passes for the committed feature contracts.
    """
    tool = _load_tool_module("validate_contracts_tool", VALIDATE_CONTRACTS_PATH)
    assert tool.validate_contracts(CONTRACTS_DIR) == []


def test_session_config_contract_tracks_runtime_enums_and_accepts_runtime_payload() -> None:
    """
    Verify the session config contract stays aligned with runtime enum values and payloads.
    """
    schema = _load_json(CONTRACTS_DIR / "session-config.schema.json")
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)

    assert schema["properties"]["environment_backend"]["enum"] == [
        backend.value for backend in EnvironmentBackend
    ]
    assert schema["properties"]["feedback_gate"]["enum"] == [
        gate.value for gate in FeedbackGate
    ]

    payload = SessionConfig(
        objective_text="maximize stability",
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=8,
        feedback_gate=FeedbackGate.ONE_REQUIRED,
        metadata={"seed": 7, "demo_enabled": True},
    ).model_dump(mode="json")

    validator.validate(payload)


def test_session_report_contract_accepts_running_exports_and_rejects_unknown_fields() -> None:
    """
    Verify the session report contract accepts running exports and rejects schema drift.
    """
    schema = _load_json(CONTRACTS_DIR / "session-report.schema.json")
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)

    assert schema["properties"]["status"]["enum"] == [status.value for status in SessionStatus]
    assert schema["properties"]["stop_reason"]["anyOf"][0]["enum"] == [
        reason.value for reason in StopReason
    ]

    payload = SessionReport(
        session_id="session-1",
        status=SessionStatus.RUNNING,
        stop_reason=None,
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=BestCandidateReport(
            candidate_id="cand-1",
            aggregate_score=0.82,
            selection_summary="Awaiting required feedback before final recommendation.",
            minor_robustness_risk_accepted=False,
        ),
        iterations=[
            IterationReport(
                iteration_index=0,
                candidate_id="cand-1",
                performance_summary="iteration 0 score=0.820 | feedback: human=1, peer=0",
                risk_level=RiskLevel.LOW,
                feedback_count=1,
            )
        ],
    ).model_dump(mode="json")

    validator.validate(payload)

    payload["unexpected"] = "drift"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)
