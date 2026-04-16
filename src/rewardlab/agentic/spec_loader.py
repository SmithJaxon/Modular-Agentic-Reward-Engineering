"""
Summary: Load and validate autonomous experiment specs from YAML or JSON files.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardlab.schemas.agent_experiment import AgentExperimentSpec


def load_experiment_spec(spec_file: Path) -> AgentExperimentSpec:
    """Load and validate an autonomous experiment spec file."""

    suffix = spec_file.suffix.strip().lower()
    raw_text = spec_file.read_text(encoding="utf-8")
    payload: dict[str, Any]

    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml(raw_text)
    elif suffix == ".json":
        payload = _load_json(raw_text)
    else:
        raise ValueError("spec file must use .yaml, .yml, or .json extension")

    return AgentExperimentSpec.model_validate(payload)


def _load_yaml(raw_text: str) -> dict[str, Any]:
    """Parse a YAML spec payload into a dictionary."""

    try:
        import yaml  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "PyYAML is required to load YAML experiment specs; use JSON or install yaml support"
        ) from exc

    parsed = yaml.safe_load(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("spec payload must be a YAML object")
    return parsed


def _load_json(raw_text: str) -> dict[str, Any]:
    """Parse a JSON spec payload into a dictionary."""

    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("spec payload must be a JSON object")
    return parsed
