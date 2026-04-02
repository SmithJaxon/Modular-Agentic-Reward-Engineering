from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import ExperimentSpecData, validate_experiment_spec_data

@dataclass(frozen=True)
class ExperimentSpec:
    """User-facing experiment specification loaded from YAML."""

    name: str
    environment: str
    baseline: str
    seed: int
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def load_experiment_spec(path: Path) -> ExperimentSpec:
    data = _load_config_payload(path.read_text(encoding="utf-8"))
    validated = validate_experiment_spec_data(data)
    return ExperimentSpec(
        name=validated.name,
        environment=validated.environment,
        baseline=validated.baseline,
        seed=validated.seed,
        notes=validated.notes,
        extra=validated.extra,
    )


def _load_config_payload(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = _parse_simple_yaml(stripped)

    if not isinstance(loaded, dict):
        raise ValueError("Experiment config must decode to a mapping")
    return loaded


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    current_section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            if not line.endswith(":"):
                key, value = _split_key_value(line)
                payload[key] = _coerce_scalar(value)
                current_section = None
            else:
                current_section = line[:-1].strip()
                payload[current_section] = {}
        else:
            if current_section is None:
                raise ValueError("Unexpected indentation in config: {0}".format(line))
            key, value = _split_key_value(line.strip())
            section = payload.get(current_section)
            if not isinstance(section, dict):
                raise ValueError("Section {0} must be a mapping".format(current_section))
            section[key] = _coerce_scalar(value)

    return payload


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError("Invalid config line: {0}".format(line))
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _coerce_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def spec_from_preset(name: str) -> ExperimentSpec:
    from .registry import get_baseline_preset

    preset = get_baseline_preset(name)
    validated = ExperimentSpecData(
        name=preset.name,
        environment=preset.environment,
        baseline=preset.baseline,
        seed=preset.seed,
        notes=preset.notes,
        extra={"source": "baseline_registry"},
    )
    return ExperimentSpec(
        name=validated.name,
        environment=validated.environment,
        baseline=validated.baseline,
        seed=validated.seed,
        notes=validated.notes,
        extra=validated.extra,
    )
