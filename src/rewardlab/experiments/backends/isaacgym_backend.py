"""
Summary: Isaac Gym backend adapter for deterministic local experiment execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from rewardlab.experiments.backends._deterministic import (
    build_artifact_ref,
    detect_runtime,
    reward_profile,
    simulate_score,
)
from rewardlab.experiments.backends.base import (
    EnvironmentBackendAdapter,
    ExperimentInput,
    ExperimentOutput,
)


class IsaacGymBackendAdapter(EnvironmentBackendAdapter):
    """
    Execute deterministic Isaac Gym performance and reflection probes.
    """

    _RUNTIME_MODULE = "isaacgym"
    _BACKEND_NAME = "isaacgym"
    _BACKEND_OFFSET = -0.02

    def run_performance(self, payload: ExperimentInput) -> ExperimentOutput:
        """
        Execute a deterministic Isaac Gym performance probe for the given payload.
        """
        runtime_available = detect_runtime(self._RUNTIME_MODULE)
        score = simulate_score(payload, backend_offset=self._BACKEND_OFFSET)
        metrics = {
            "backend": self._BACKEND_NAME,
            "variant_label": payload.variant_label,
            "seed": payload.seed,
            "overrides": payload.overrides,
            "reward_profile": reward_profile(payload.reward_definition),
            "runtime_available": runtime_available,
            "score": score,
        }
        return ExperimentOutput(
            score=score,
            metrics=metrics,
            summary=(
                f"{self._BACKEND_NAME} performance variant={payload.variant_label} "
                f"iteration={payload.iteration_index} score={score:.3f}"
            ),
            artifact_refs=(build_artifact_ref(self._BACKEND_NAME, payload, "performance"),),
        )

    def run_reflection(self, payload: ExperimentInput) -> ExperimentOutput:
        """
        Execute a deterministic reflection-style pass for the given payload.
        """
        runtime_available = detect_runtime(self._RUNTIME_MODULE)
        score = max(
            0.05,
            round(simulate_score(payload, backend_offset=self._BACKEND_OFFSET) - 0.025, 4),
        )
        metrics = {
            "backend": self._BACKEND_NAME,
            "variant_label": payload.variant_label,
            "runtime_available": runtime_available,
            "score": score,
        }
        return ExperimentOutput(
            score=score,
            metrics=metrics,
            summary=(
                f"{self._BACKEND_NAME} reflection iteration={payload.iteration_index}: "
                "probe hardware-robust shaping adjustments"
            ),
            artifact_refs=(build_artifact_ref(self._BACKEND_NAME, payload, "reflection"),),
        )
