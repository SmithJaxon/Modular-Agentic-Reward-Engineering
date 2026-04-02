"""
Summary: Deterministic backend simulation helpers for local adapter execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from importlib import import_module

from rewardlab.experiments.backends.base import ExperimentInput

_BASE_VARIANT_PENALTIES = {
    "default": 0.0,
    "observation_dropout": 0.04,
    "dynamics_shift": 0.07,
    "reward_delay": 0.05,
}
_PROFILE_VARIANT_PENALTIES = {
    "stable": {
        "default": 0.0,
        "observation_dropout": 0.01,
        "dynamics_shift": 0.02,
        "reward_delay": 0.01,
    },
    "fragile": {
        "default": 0.0,
        "observation_dropout": 0.05,
        "dynamics_shift": 0.09,
        "reward_delay": 0.06,
    },
    "exploit": {
        "default": 0.0,
        "observation_dropout": 0.20,
        "dynamics_shift": 0.28,
        "reward_delay": 0.24,
    },
}


def detect_runtime(module_name: str) -> bool:
    """
    Probe whether an optional backend runtime can be imported.
    """
    try:
        import_module(module_name)
    except ImportError:
        return False
    return True


def reward_profile(reward_definition: str) -> str:
    """
    Classify reward text into a deterministic robustness profile bucket.
    """
    normalized = reward_definition.lower()
    if "exploit" in normalized or "hack" in normalized:
        return "exploit"
    if "fragile" in normalized or "narrow" in normalized:
        return "fragile"
    return "stable"


def simulate_score(payload: ExperimentInput, backend_offset: float) -> float:
    """
    Compute a deterministic backend score for the provided experiment input.
    """
    profile = reward_profile(payload.reward_definition)
    base = 0.56 + (0.07 * payload.iteration_index) + backend_offset
    if "stability" in payload.reward_definition.lower():
        base += 0.03
    if "smooth" in payload.reward_definition.lower():
        base += 0.02
    if "speed" in payload.reward_definition.lower():
        base += 0.04
    if profile == "fragile":
        base += 0.05
    if profile == "exploit":
        base += 0.10

    variant_penalty = _BASE_VARIANT_PENALTIES.get(payload.variant_label, 0.06)
    variant_penalty += _PROFILE_VARIANT_PENALTIES[profile].get(payload.variant_label, 0.04)

    seed_shift = (payload.seed % 5) * 0.003
    score = base - variant_penalty - seed_shift
    return round(min(0.99, max(0.05, score)), 4)


def build_artifact_ref(backend_name: str, payload: ExperimentInput, suffix: str) -> str:
    """
    Construct a deterministic artifact reference string for one simulated run.
    """
    return (
        f"{backend_name}/{payload.environment_id}/{payload.session_id}/"
        f"{payload.variant_label}/{suffix}-{payload.iteration_index}.json"
    )
