from __future__ import annotations

import random

from pathlib import Path

from .contracts import EnvironmentAdapter, RunArtifact, RunReport
from .manifest import ExperimentManifest


class PlaceholderIsaacGymAdapter:
    """Temporary adapter used until Phase 2 wires in IsaacGym."""

    environment_name = "placeholder"

    def train(self, manifest: ExperimentManifest) -> RunReport:
        random.seed(manifest.seed)
        return RunReport(
            manifest=manifest,
            status="trained_placeholder",
            metrics={
                "seed": float(manifest.seed),
                "training_score": random.random(),
            },
            warnings=["IsaacGym adapter not yet implemented"],
        )

    def evaluate(self, manifest: ExperimentManifest) -> RunReport:
        random.seed(manifest.seed + 1)
        return RunReport(
            manifest=manifest,
            status="evaluated_placeholder",
            metrics={
                "seed": float(manifest.seed),
                "evaluation_score": random.random(),
            },
            warnings=["IsaacGym adapter not yet implemented"],
        )

    def summarize(self, report: RunReport) -> str:
        score = report.metrics.get("evaluation_score")
        if score is None:
            score = report.metrics.get("training_score")
        return "status={0} score={1}".format(report.status, score)


def is_environment_adapter(candidate: object) -> bool:
    return isinstance(candidate, EnvironmentAdapter)
