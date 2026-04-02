from __future__ import annotations

import random

from .contracts import RunReport
from .environments import AllegroHandAdapter, CartPoleAdapter, HumanoidAdapter
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


def get_environment_adapter(environment_name: str):
    if environment_name == "CartPole":
        return CartPoleAdapter()
    if environment_name == "Humanoid":
        return HumanoidAdapter()
    if environment_name == "AllegroHand":
        return AllegroHandAdapter()
    raise KeyError("Unknown environment name: {0}".format(environment_name))
