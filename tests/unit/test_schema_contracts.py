"""
Summary: Regression tests ensuring session schema payloads remain contract-compatible.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.schemas.session_report import (
    IterationSummary,
    ReportStatus,
    RiskLevel,
    SelectionCandidate,
    SessionReport,
)

CONTRACTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "001-iterative-reward-design"
    / "contracts"
)


def load_schema(name: str) -> dict[str, object]:
    """Load a JSON schema document from the checked-in contract directory."""

    return json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8-sig"))


def test_session_config_payload_matches_contract_schema() -> None:
    """The runtime session config model should still validate against the contract schema."""

    schema = load_schema("session-config.schema.json")
    payload = SessionConfig(
        objective_text="Reward smooth balance and centered control.",
        environment_id="cartpole-v1",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        no_improve_limit=3,
        max_iterations=5,
        feedback_gate=FeedbackGate.NONE,
        metadata={"operator_seed": 7},
    ).model_dump(mode="json")

    Draft202012Validator(schema).validate(payload)


def test_session_report_payload_matches_contract_schema() -> None:
    """The runtime session report model should still validate against the contract schema."""

    schema = load_schema("session-report.schema.json")
    payload = SessionReport(
        session_id="session-001",
        status=ReportStatus.INTERRUPTED,
        stop_reason="user_interrupt",
        environment_backend=EnvironmentBackend.GYMNASIUM,
        best_candidate=SelectionCandidate(
            candidate_id="candidate-001",
            aggregate_score=1.7,
            selection_summary="Feedback gate satisfied. Highest-ranked candidate selected.",
        ),
        iterations=[
            IterationSummary(
                iteration_index=0,
                candidate_id="candidate-000",
                performance_summary="Baseline candidate initialized.",
                risk_level=RiskLevel.LOW,
                feedback_count=0,
            )
        ],
    ).model_dump(mode="json")

    Draft202012Validator(schema).validate(payload)
