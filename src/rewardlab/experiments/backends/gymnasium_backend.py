"""
Summary: Gymnasium backend adapter for deterministic or PPO-backed local experiments.
Created: 2026-04-02
Last Updated: 2026-04-06
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(slots=True)
class _CachedRun:
    """
    Hold the last real Gymnasium execution so reflection can reuse it.
    """

    payload: ExperimentInput
    result: Any


class GymnasiumBackendAdapter(EnvironmentBackendAdapter):
    """
    Execute deterministic Gymnasium probes or PPO-backed real runs.
    """

    _RUNTIME_MODULE = "gymnasium"
    _BACKEND_NAME = "gymnasium"
    _BACKEND_OFFSET = 0.0

    def __init__(self) -> None:
        """
        Initialize the adapter cache for real PPO-backed executions.
        """
        self._cached_run: _CachedRun | None = None

    def run_performance(self, payload: ExperimentInput) -> ExperimentOutput:
        """
        Execute a Gymnasium performance probe for the given payload.
        """
        if self._is_real_execution(payload):
            real_runtime = _load_real_runtime()
            real_result = real_runtime.run_gymnasium_ppo_experiment(payload)
            self._cached_run = _CachedRun(payload=payload, result=real_result)
            return ExperimentOutput(
                score=real_result.score,
                metrics=real_result.metrics,
                summary=real_result.performance_summary,
                artifact_refs=real_result.artifacts.refs(),
            )

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
        Execute a Gymnasium reflection probe for the given payload.
        """
        if self._is_real_execution(payload):
            if self._cached_run is None or self._cached_run.payload != payload:
                real_runtime = _load_real_runtime()
                real_result = real_runtime.run_gymnasium_ppo_experiment(payload)
                self._cached_run = _CachedRun(payload=payload, result=real_result)
            assert self._cached_run is not None
            metrics = dict(self._cached_run.result.metrics)
            metrics["reflection_generated"] = True
            return ExperimentOutput(
                score=self._cached_run.result.score,
                metrics=metrics,
                summary=self._cached_run.result.reflection_summary,
                artifact_refs=self._cached_run.result.artifacts.refs(),
            )

        runtime_available = detect_runtime(self._RUNTIME_MODULE)
        score = max(
            0.05,
            round(simulate_score(payload, backend_offset=self._BACKEND_OFFSET) - 0.02, 4),
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
                "prefer stability-preserving revisions"
            ),
            artifact_refs=(build_artifact_ref(self._BACKEND_NAME, payload, "reflection"),),
        )

    @staticmethod
    def _is_real_execution(payload: ExperimentInput) -> bool:
        """
        Report whether the payload requests PPO-backed runtime execution.
        """
        real_runtime = _load_real_runtime()
        config = real_runtime.GymnasiumExecutionConfig.from_overrides(
            payload.overrides,
            variant_label=payload.variant_label,
        )
        return bool(config.is_real_execution)


def _load_real_runtime() -> Any:
    """
    Import the optional real Gymnasium runtime helpers on demand.
    """
    from rewardlab.experiments import gymnasium_runtime

    return gymnasium_runtime
