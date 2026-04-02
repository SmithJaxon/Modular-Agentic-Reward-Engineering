from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .schema import ALLOWED_ENVIRONMENTS


@dataclass(frozen=True)
class BaselinePreset:
    """Canonical baseline metadata for a target environment."""

    name: str
    environment: str
    baseline: str
    seed: int
    notes: str


BASELINE_REGISTRY: Dict[str, BaselinePreset] = {
    "cartpole": BaselinePreset(
        name="cartpole_ppo_baseline",
        environment="CartPole",
        baseline="PPO",
        seed=7,
        notes="Baseline preset for CartPole Phase 1 scaffold",
    ),
    "humanoid": BaselinePreset(
        name="humanoid_ppo_baseline",
        environment="Humanoid",
        baseline="PPO",
        seed=11,
        notes="Baseline preset for Humanoid Phase 1 scaffold",
    ),
    "allegro_hand": BaselinePreset(
        name="allegro_hand_ppo_baseline",
        environment="AllegroHand",
        baseline="PPO",
        seed=13,
        notes="Baseline preset for Allegro Hand Phase 1 scaffold",
    ),
}


def get_baseline_preset(name: str) -> BaselinePreset:
    key = name.strip().lower()
    if key not in BASELINE_REGISTRY:
        raise KeyError("Unknown baseline preset: {0}".format(name))
    return BASELINE_REGISTRY[key]


def list_baseline_presets() -> List[BaselinePreset]:
    return [BASELINE_REGISTRY[key] for key in sorted(BASELINE_REGISTRY)]


def validate_registry() -> None:
    environments = {preset.environment for preset in BASELINE_REGISTRY.values()}
    missing = [env for env in ALLOWED_ENVIRONMENTS if env not in environments]
    if missing:
        raise ValueError("Missing baseline presets for: " + ", ".join(missing))

