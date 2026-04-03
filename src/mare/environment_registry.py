from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .environment import IsaacGymEnvironmentProfile


@dataclass(frozen=True)
class EnvironmentPreset:
    """Bundle of a profile and human-readable notes."""

    profile: IsaacGymEnvironmentProfile
    notes: str


ENVIRONMENT_PRESETS: Dict[str, EnvironmentPreset] = {
    "CartPole": EnvironmentPreset(
        profile=IsaacGymEnvironmentProfile(
            name="CartPole",
            task_name="cartpole",
            default_train_steps=50_000,
            default_eval_episodes=10,
            default_device="auto",
            default_hyperparameters={
                "gamma": 0.99,
                "learning_rate": 3e-4,
                "clip_range": 0.2,
            },
        ),
        notes="Lightweight control task used for fast iteration.",
    ),
    "Humanoid": EnvironmentPreset(
        profile=IsaacGymEnvironmentProfile(
            name="Humanoid",
            task_name="humanoid",
            default_train_steps=250_000,
            default_eval_episodes=5,
            default_device="cuda",
            default_hyperparameters={
                "gamma": 0.99,
                "learning_rate": 3e-4,
                "clip_range": 0.2,
            },
        ),
        notes="Higher-complexity locomotion benchmark.",
    ),
    "AllegroHand": EnvironmentPreset(
        profile=IsaacGymEnvironmentProfile(
            name="AllegroHand",
            task_name="allegro_hand",
            default_train_steps=250_000,
            default_eval_episodes=5,
            default_device="cuda",
            default_hyperparameters={
                "gamma": 0.99,
                "learning_rate": 3e-4,
                "clip_range": 0.2,
            },
        ),
        notes="Dexterous manipulation benchmark.",
    ),
}


def get_environment_preset(name: str) -> EnvironmentPreset:
    if name not in ENVIRONMENT_PRESETS:
        raise KeyError("Unknown environment preset: {0}".format(name))
    return ENVIRONMENT_PRESETS[name]


def list_environment_presets() -> List[EnvironmentPreset]:
    return [ENVIRONMENT_PRESETS[key] for key in sorted(ENVIRONMENT_PRESETS)]
