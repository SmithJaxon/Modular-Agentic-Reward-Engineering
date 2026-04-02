"""
Summary: Probe-matrix robustness runner for RewardLab reward candidates.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rewardlab.experiments.backends.base import BackendAdapter
from rewardlab.experiments.backends.factory import resolve_backend
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
        gymnasium_backend: BackendAdapter | None = None,
        isaacgym_backend: BackendAdapter | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
    ) -> None:
        """Store the probe matrix location and backend dependencies."""

        self.probe_matrix_path = probe_matrix_path
        self.gymnasium_backend = gymnasium_backend
        self.isaacgym_backend = isaacgym_backend
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()

    def run_candidate_probes(
        self,
        *,
        candidate: RewardCandidate,
        environment_backend: EnvironmentBackend,
        environment_id: str,
        policy: Callable[[Any], Any],
    ) -> tuple[list[ExperimentRun], RobustnessAssessment]:
        """Execute all configured robustness probes for a candidate."""

        del environment_id

        adapter = resolve_backend(
            environment_backend,
            gymnasium_backend=self.gymnasium_backend,
            isaacgym_backend=self.isaacgym_backend,
        )
        variants = self._load_variants(environment_backend.value)
        runs: list[ExperimentRun] = []
        for index, variant in enumerate(variants, start=1):
            started_at = datetime.now(UTC)
            episode = adapter.run_episode(
                variant.environment_id,
                policy=policy,
                seed=variant.seed,
            )
            ended_at = datetime.now(UTC)
            runs.append(
                ExperimentRun(
                    run_id=f"{candidate.candidate_id}-robustness-{index:03d}",
                    candidate_id=candidate.candidate_id,
                    run_type=RunType.ROBUSTNESS,
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

        assessment = self.risk_analyzer.assess_candidate(
            candidate=candidate,
            runs=runs,
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
