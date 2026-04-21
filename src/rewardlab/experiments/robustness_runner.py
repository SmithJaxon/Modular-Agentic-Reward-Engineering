"""
Summary: Probe-matrix robustness runner for RewardLab reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rewardlab.experiments.backends.base import BackendAdapter
from rewardlab.experiments.backends.factory import resolve_backend
from rewardlab.experiments.execution_service import (
    ExecutionOutcome,
    ExecutionRequest,
    ExperimentExecutionService,
    ExperimentRunner,
)
from rewardlab.schemas.experiment_run import ExperimentRun, RunStatus, RunType
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.robustness_assessment import RobustnessAssessment
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.selection.risk_analyzer import RiskAnalyzer


@dataclass(frozen=True)
class ProbeVariant:
    """One robustness probe variant loaded from the probe matrix."""

    label: str
    environment_id: str
    seed: int | None
    overrides: dict[str, float | int]


class RobustnessRunner:
    """Run candidate robustness probes through the selected backend adapter."""

    def __init__(
        self,
        *,
        probe_matrix_path: Path,
        experiment_execution_service: ExperimentExecutionService | None = None,
        gymnasium_runner: ExperimentRunner | None = None,
        isaacgym_runner: ExperimentRunner | None = None,
        gymnasium_backend: BackendAdapter | None = None,
        isaacgym_backend: BackendAdapter | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
    ) -> None:
        """Store the probe matrix location and backend dependencies."""

        self.probe_matrix_path = probe_matrix_path
        self.experiment_execution_service = experiment_execution_service
        self.gymnasium_runner = gymnasium_runner
        self.isaacgym_runner = isaacgym_runner
        self.gymnasium_backend = gymnasium_backend
        self.isaacgym_backend = isaacgym_backend
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()

    def run_candidate_probes(
        self,
        *,
        candidate: RewardCandidate,
        primary_run: ExperimentRun,
        environment_backend: EnvironmentBackend,
        environment_id: str,
        policy: Callable[[Any], Any] | None = None,
    ) -> tuple[list[ExperimentRun], RobustnessAssessment | None]:
        """Execute all configured robustness probes for a candidate."""

        del environment_id

        variants = self._load_variants(environment_backend.value)
        if self.experiment_execution_service is not None:
            runs = self._execute_actual_backend_probes(
                candidate=candidate,
                primary_run=primary_run,
                environment_backend=environment_backend,
                variants=variants,
            )
        else:
            if policy is None:
                raise ValueError("policy is required when no execution service is configured")
            runs = self._execute_adapter_probes(
                candidate=candidate,
                primary_run=primary_run,
                environment_backend=environment_backend,
                variants=variants,
                policy=policy,
            )

        completed_runs = [run for run in runs if run.status == RunStatus.COMPLETED]
        if not completed_runs:
            return runs, None

        assessment = self.risk_analyzer.assess_candidate(
            candidate=candidate,
            primary_run=primary_run,
            runs=completed_runs,
        )
        if len(completed_runs) != len(runs):
            failed_count = len(runs) - len(completed_runs)
            assessment = assessment.model_copy(
                update={
                    "risk_notes": (
                        f"{assessment.risk_notes} "
                        f"{failed_count} probe variant(s) failed."
                    )
                }
            )
        return runs, assessment

    def _load_variants(self, backend_name: str) -> list[ProbeVariant]:
        """Load probe variants for the requested backend from the matrix file."""

        document = json.loads(self.probe_matrix_path.read_text(encoding="utf-8"))
        backend_entries = document["backends"][backend_name]
        return [
            ProbeVariant(
                label=str(entry["label"]),
                environment_id=str(entry["environment_id"]),
                seed=int(entry["seed"]) if entry.get("seed") is not None else None,
                overrides=_coerce_overrides(entry.get("overrides", {})),
            )
            for entry in backend_entries
        ]

    def _execute_actual_backend_probes(
        self,
        *,
        candidate: RewardCandidate,
        primary_run: ExperimentRun,
        environment_backend: EnvironmentBackend,
        variants: list[ProbeVariant],
    ) -> list[ExperimentRun]:
        """Execute probe variants through the shared real-execution service."""

        assert self.experiment_execution_service is not None
        runner = self._resolve_execution_runner(environment_backend)
        runs: list[ExperimentRun] = []
        for index, variant in enumerate(variants, start=1):
            result = self.experiment_execution_service.execute_candidate(
                candidate=candidate,
                request=ExecutionRequest(
                    run_id=f"{candidate.candidate_id}-robustness-{index:03d}",
                    backend=environment_backend,
                    environment_id=variant.environment_id,
                    run_type=RunType.ROBUSTNESS,
                    execution_mode=primary_run.execution_mode,
                    variant_label=variant.label,
                    seed=variant.seed,
                ),
                runner=_probe_execution_runner(
                    base_runner=runner,
                    variant=variant,
                    primary_run_id=primary_run.run_id,
                ),
            )
            runs.append(result.run)
        return runs

    def _execute_adapter_probes(
        self,
        *,
        candidate: RewardCandidate,
        primary_run: ExperimentRun,
        environment_backend: EnvironmentBackend,
        variants: list[ProbeVariant],
        policy: Callable[[Any], Any],
    ) -> list[ExperimentRun]:
        """Execute probe variants directly through a backend adapter for offline tests."""

        adapter = resolve_backend(
            environment_backend,
            gymnasium_backend=self.gymnasium_backend,
            isaacgym_backend=self.isaacgym_backend,
        )
        runs: list[ExperimentRun] = []
        for index, variant in enumerate(variants, start=1):
            started_at = datetime.now(timezone.utc)
            episode = adapter.run_episode(
                variant.environment_id,
                policy=policy,
                seed=variant.seed,
            )
            ended_at = datetime.now(timezone.utc)
            runs.append(
                ExperimentRun(
                    run_id=f"{candidate.candidate_id}-robustness-{index:03d}",
                    candidate_id=candidate.candidate_id,
                    backend=environment_backend,
                    environment_id=variant.environment_id,
                    run_type=RunType.ROBUSTNESS,
                    execution_mode=primary_run.execution_mode,
                    variant_label=variant.label,
                    seed=variant.seed,
                    status=RunStatus.COMPLETED,
                    metrics={
                        "total_reward": episode.total_reward,
                        "step_count": len(episode.steps),
                        "overrides": variant.overrides,
                    },
                    started_at=started_at,
                    ended_at=ended_at,
                )
            )
        return runs

    def _resolve_execution_runner(
        self,
        environment_backend: EnvironmentBackend,
    ) -> ExperimentRunner:
        """Return the configured execution runner for the requested backend."""

        if environment_backend == EnvironmentBackend.GYMNASIUM:
            if self.gymnasium_runner is None:
                raise RuntimeError("gymnasium_runner is required for actual robustness probes")
            return self.gymnasium_runner
        if environment_backend == EnvironmentBackend.ISAAC_GYM:
            if self.isaacgym_runner is None:
                raise RuntimeError("isaacgym_runner is required for actual robustness probes")
            return self.isaacgym_runner
        raise ValueError(f"unsupported environment backend: {environment_backend.value!r}")


def _coerce_overrides(raw_overrides: Any) -> dict[str, float | int]:
    """Validate and normalize numeric override values from the probe matrix."""

    if not isinstance(raw_overrides, dict):
        raise ValueError("probe overrides must be a JSON object")
    normalized: dict[str, float | int] = {}
    for key, value in raw_overrides.items():
        if not isinstance(key, str):
            raise ValueError("probe override keys must be strings")
        if not isinstance(value, (int, float)):
            raise ValueError("probe override values must be numeric")
        normalized[key] = value
    return normalized


def _probe_execution_runner(
    *,
    base_runner: ExperimentRunner,
    variant: ProbeVariant,
    primary_run_id: str,
) -> ExperimentRunner:
    """Wrap a backend runner with robustness-specific manifest and metric metadata."""

    def run_probe(
        execution_request: ExecutionRequest,
        reward_program: Any,
    ) -> ExecutionOutcome:
        """Execute one probe rollout and annotate its emitted evidence."""

        outcome = base_runner(execution_request, reward_program)
        metrics = dict(outcome.metrics)
        if variant.overrides:
            metrics["probe_overrides"] = dict(variant.overrides)
        manifest_metadata = {
            **outcome.manifest_metadata,
            "primary_run_id": primary_run_id,
            "probe_variant": variant.label,
            "probe_overrides": dict(variant.overrides),
        }
        return ExecutionOutcome(
            metrics=metrics,
            event_trace=outcome.event_trace,
            runtime_status=outcome.runtime_status,
            manifest_metadata=manifest_metadata,
        )

    return run_probe

