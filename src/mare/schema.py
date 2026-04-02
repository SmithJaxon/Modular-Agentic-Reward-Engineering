from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


ALLOWED_ENVIRONMENTS = ("CartPole", "Humanoid", "AllegroHand")
ALLOWED_BASELINES = ("PPO",)


@dataclass(frozen=True)
class ExperimentSpecData:
    """Validated experiment specification payload."""

    name: str
    environment: str
    baseline: str
    seed: int
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def validate_experiment_spec_data(data: Dict[str, Any]) -> ExperimentSpecData:
    missing = [key for key in ("name", "environment", "baseline") if key not in data]
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))

    environment = str(data["environment"])
    baseline = str(data["baseline"])
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(
            "Unsupported environment: {0}. Allowed: {1}".format(
                environment, ", ".join(ALLOWED_ENVIRONMENTS)
            )
        )
    if baseline not in ALLOWED_BASELINES:
        raise ValueError(
            "Unsupported baseline: {0}. Allowed: {1}".format(
                baseline, ", ".join(ALLOWED_BASELINES)
            )
        )

    seed = int(data.get("seed", 0))
    if seed < 0:
        raise ValueError("seed must be non-negative")

    extra = data.get("extra") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra must be a mapping")

    return ExperimentSpecData(
        name=str(data["name"]),
        environment=environment,
        baseline=baseline,
        seed=seed,
        notes=data.get("notes"),
        extra=dict(extra),
    )

