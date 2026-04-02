from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

from ..launch import LaunchTarget, PPORunContract
from ..contracts import RunReport
from ..environment import IsaacGymEnvironmentProfile, IsaacGymExecutionPlan
from ..manifest import ExperimentManifest


@dataclass(frozen=True)
class BaseIsaacGymAdapter(ABC):
    """Base class for IsaacGym-backed environment adapters."""

    profile: IsaacGymEnvironmentProfile

    @property
    def environment_name(self) -> str:
        return self.profile.name

    def build_training_plan(self, manifest: ExperimentManifest) -> IsaacGymExecutionPlan:
        return IsaacGymExecutionPlan(
            environment=self.profile.name,
            task_name=self.profile.task_name,
            algorithm=manifest.baseline,
            seed=manifest.seed,
            train_steps=self.profile.default_train_steps,
            eval_episodes=self.profile.default_eval_episodes,
            hyperparameters=dict(self.profile.default_hyperparameters),
        )

    def build_evaluation_plan(self, manifest: ExperimentManifest) -> IsaacGymExecutionPlan:
        return self.build_training_plan(manifest)

    def build_run_contract(
        self,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> PPORunContract:
        return PPORunContract(
            manifest=manifest,
            execution_plan=self.build_training_plan(manifest),
            run_dir=run_dir,
            launch_target=launch_target,
        )

    def train(self, manifest: ExperimentManifest) -> RunReport:
        plan = self.build_training_plan(manifest)
        return self._placeholder_report(manifest, plan, status="training_planned")

    def evaluate(self, manifest: ExperimentManifest) -> RunReport:
        plan = self.build_evaluation_plan(manifest)
        return self._placeholder_report(manifest, plan, status="evaluation_planned")

    def summarize(self, report: RunReport) -> str:
        metrics = ", ".join(f"{k}={v}" for k, v in sorted(report.metrics.items()))
        return "{0}: {1}".format(report.status, metrics or "no-metrics")

    def _placeholder_report(
        self,
        manifest: ExperimentManifest,
        plan: IsaacGymExecutionPlan,
        status: str,
    ) -> RunReport:
        metrics = {
            "seed": float(manifest.seed),
            "train_steps": float(plan.train_steps),
            "eval_episodes": float(plan.eval_episodes),
        }
        return RunReport(
            manifest=manifest,
            status=status,
            metrics=metrics,
            warnings=[
                "IsaacGym runtime not yet connected; plan generated only"
            ],
            artifacts=[],
            notes=plan.to_dict().__repr__(),
        )
