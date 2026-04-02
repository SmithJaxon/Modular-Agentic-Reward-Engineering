"""
Summary: Robustness probe execution and aggregation for reward-hacking detection.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.backends.factory import resolve_backend_adapter
from rewardlab.schemas.experiment_run import (
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentRunType,
)
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.selection.risk_analyzer import RiskAnalysisResult, RiskAnalyzer

_DEFAULT_PROBE_MATRIX = (
    Path(__file__).resolve().parents[3] / "tools" / "reward_hack_probes" / "probe_matrix.yaml"
)


@dataclass(slots=True, frozen=True)
class ProbeVariant:
    """
    Represent one configured robustness probe variant.
    """

    variant_label: str
    seed: int
    overrides: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RobustnessRunResult:
    """
    Bundle executed robustness runs with their derived risk analysis.
    """

    experiment_runs: list[ExperimentRun]
    analysis: RiskAnalysisResult

    @property
    def assessment(self) -> RobustnessAssessment:
        """
        Expose the derived robustness assessment directly on the result.
        """
        return self.analysis.assessment


class RobustnessRunner:
    """
    Execute configured probe variants and derive candidate robustness assessments.
    """

    def __init__(
        self,
        probe_matrix_path: Path | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
    ) -> None:
        """
        Initialize the runner with probe configuration and analysis dependencies.

        Args:
            probe_matrix_path: Optional probe matrix override path.
            risk_analyzer: Optional risk analyzer override.
        """
        self._probe_matrix_path = probe_matrix_path or _DEFAULT_PROBE_MATRIX
        self._risk_analyzer = risk_analyzer or RiskAnalyzer()

    def run(
        self,
        candidate_id: str,
        payload: ExperimentInput,
        primary_score: float,
    ) -> RobustnessRunResult:
        """
        Execute all configured robustness probes for the payload backend.

        Args:
            candidate_id: Candidate identifier under evaluation.
            payload: Primary experiment input payload.
            primary_score: Score from the primary performance run.

        Returns:
            Robustness run bundle with executed probe runs and derived analysis.
        """
        adapter = resolve_backend_adapter(payload.environment_backend)
        timestamp = datetime.now(UTC).isoformat()
        runs: list[ExperimentRun] = []
        for variant in self._variants_for_backend(payload.environment_backend.value):
            run_payload = ExperimentInput(
                session_id=payload.session_id,
                environment_id=payload.environment_id,
                environment_backend=payload.environment_backend,
                reward_definition=payload.reward_definition,
                iteration_index=payload.iteration_index,
                objective_text=payload.objective_text,
                variant_label=variant.variant_label,
                seed=variant.seed,
                overrides=variant.overrides,
            )
            output = adapter.run_performance(run_payload)
            runs.append(
                ExperimentRun(
                    run_id=f"run-{uuid4().hex[:12]}",
                    candidate_id=candidate_id,
                    run_type=ExperimentRunType.ROBUSTNESS,
                    variant_label=variant.variant_label,
                    seed=variant.seed,
                    status=ExperimentRunStatus.COMPLETED,
                    metrics=output.metrics,
                    artifact_refs=list(output.artifact_refs),
                    started_at=timestamp,
                    ended_at=timestamp,
                )
            )
        analysis = self._risk_analyzer.analyze(
            candidate_id=candidate_id,
            primary_score=primary_score,
            robustness_runs=runs,
        )
        return RobustnessRunResult(experiment_runs=runs, analysis=analysis)

    def _variants_for_backend(self, backend_name: str) -> list[ProbeVariant]:
        """
        Resolve configured probe variants for the requested backend.
        """
        raw = self._load_probe_matrix()
        entries = raw.get(backend_name, [])
        return [
            ProbeVariant(
                variant_label=str(entry["variant_label"]),
                seed=int(entry["seed"]),
                overrides=dict(entry.get("overrides", {})),
            )
            for entry in entries
        ]

    def _load_probe_matrix(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load the probe matrix from JSON or the repository's YAML subset.
        """
        text = self._probe_matrix_path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = _parse_probe_matrix_yaml(text)
        if not isinstance(raw, dict):
            raise ValueError("probe matrix must be a top-level object")
        normalized: dict[str, list[dict[str, Any]]] = {}
        for backend_name, entries in raw.items():
            if not isinstance(entries, list):
                raise ValueError(f"probe matrix entries for {backend_name} must be a list")
            normalized[str(backend_name)] = [dict(entry) for entry in entries]
        return normalized


def _parse_probe_matrix_yaml(text: str) -> dict[str, list[dict[str, Any]]]:
    """
    Parse the constrained YAML shape used by the repository probe matrix.

    Supported structure:
    - top-level backend mappings
    - per-backend lists of probe dictionaries
    - nested `overrides` dictionaries
    - comments, blank lines, quoted or unquoted scalars
    """
    result: dict[str, list[dict[str, Any]]] = {}
    current_backend: str | None = None
    current_entry: dict[str, Any] | None = None
    current_nested_key: str | None = None

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        if indent == 0:
            if not content.endswith(":"):
                raise ValueError(f"invalid top-level probe matrix line: {content}")
            current_backend = content[:-1].strip()
            result[current_backend] = []
            current_entry = None
            current_nested_key = None
            continue

        if indent == 2 and content.startswith("- "):
            if current_backend is None:
                raise ValueError("probe variant defined before backend section")
            current_entry = {}
            result[current_backend].append(current_entry)
            current_nested_key = None
            item_content = content[2:].strip()
            if item_content:
                key, value = _split_mapping(item_content)
                current_entry[key] = _parse_scalar(value)
            continue

        if indent == 4:
            if current_entry is None:
                raise ValueError(f"probe field defined before list item: {content}")
            key, value = _split_mapping(content)
            if value == "":
                current_entry[key] = {}
                current_nested_key = key
            else:
                current_entry[key] = _parse_scalar(value)
                current_nested_key = None
            continue

        if indent == 6:
            if current_entry is None or current_nested_key is None:
                raise ValueError(f"nested override field has no parent mapping: {content}")
            key, value = _split_mapping(content)
            nested = current_entry.get(current_nested_key)
            if not isinstance(nested, dict):
                raise ValueError(f"nested key {current_nested_key} must map to an object")
            nested[key] = _parse_scalar(value)
            continue

        raise ValueError(f"unsupported indentation in probe matrix: {content}")

    return result


def _split_mapping(content: str) -> tuple[str, str]:
    """
    Split one YAML `key: value` mapping entry.
    """
    key, separator, value = content.partition(":")
    if separator == "":
        raise ValueError(f"invalid mapping entry: {content}")
    return key.strip(), value.strip()


def _strip_comment(line: str) -> str:
    """
    Remove YAML comments while preserving quoted scalar content.
    """
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    """
    Parse a YAML scalar into a Python value for the probe matrix.
    """
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
