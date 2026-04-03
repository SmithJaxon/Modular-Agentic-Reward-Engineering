from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .experiment import ExperimentRunner
from .launch import LaunchTarget
from .manifest import ExperimentManifest
from .robustness import RewardRobustnessAnalyzer, RobustnessAssessment


@dataclass(frozen=True)
class SweepVariantSpec:
    """Template for a planned sweep variant."""

    suffix: str
    rationale: str
    seed_offset: int = 0
    baseline_override: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass(frozen=True)
class SweepRun:
    """A planned future run variant."""

    name: str
    run_dir: Path
    manifest: ExperimentManifest
    launch_target: LaunchTarget
    command: List[str]
    rationale: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": str(self.run_dir),
            "manifest": asdict(self.manifest),
            "launch_target": self.launch_target.to_dict(),
            "command": self.command,
            "rationale": self.rationale,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SweepPlan:
    """Compact plan for a future VM experiment sweep."""

    source: str
    environment: str
    base_run_dir: Path
    assessment: Optional[Dict[str, Any]]
    static_checks: List[Dict[str, Any]] = field(default_factory=list)
    runs: List[SweepRun] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "environment": self.environment,
            "base_run_dir": str(self.base_run_dir),
            "assessment": self.assessment,
            "static_checks": self.static_checks,
            "runs": [run.to_dict() for run in self.runs],
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


class SweepPlanner:
    """Build a cheap, traceable sweep plan for future VM runs."""

    def __init__(
        self,
        runner: Optional[ExperimentRunner] = None,
        analyzer: Optional[RewardRobustnessAnalyzer] = None,
    ) -> None:
        self.runner = runner
        self.analyzer = analyzer or RewardRobustnessAnalyzer()

    def build_from_manifest(
        self,
        manifest: ExperimentManifest,
        base_run_dir: Path,
        launch_target: LaunchTarget,
        reward_candidate_path: Optional[Path] = None,
        reward_entrypoint: str = "compute_reward",
        max_runs: int = 3,
    ) -> SweepPlan:
        assessment = self._assess_candidate(reward_candidate_path, reward_entrypoint)
        return self._build_plan(
            manifest=manifest,
            base_run_dir=base_run_dir,
            launch_target=launch_target,
            assessment=assessment,
            source="manifest",
            max_runs=max_runs,
        )

    def build_from_trace(
        self,
        trace_path: Path,
        base_run_dir: Optional[Path] = None,
        launch_target: Optional[LaunchTarget] = None,
        max_runs: int = 3,
    ) -> SweepPlan:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest_data = trace_data.get("manifest", {})
        manifest = ExperimentManifest(
            name=str(manifest_data.get("name", "trace_sweep")),
            environment=str(manifest_data.get("environment", "CartPole")),
            baseline=str(manifest_data.get("baseline", "PPO")),
            seed=int(manifest_data.get("seed", 0)),
            created_at=str(manifest_data.get("created_at", "")),
            notes=manifest_data.get("notes"),
            extra=dict(manifest_data.get("extra") or {}),
            reward_candidate=manifest_data.get("reward_candidate") or trace_data.get("reward_candidate"),
        )
        candidate = manifest.reward_candidate or {}
        candidate_path = candidate.get("path")
        reward_path = Path(candidate_path) if candidate_path else None
        assessment = self.analyzer.assess_trace(trace_path) if reward_path is not None else None
        if base_run_dir is None:
            base_run_dir = trace_path.parent
        if launch_target is None:
            launch_target = LaunchTarget(
                kind="gpu_vm",
                python_executable="python3",
                working_directory=self._trace_project_root(trace_path),
                gpu_required=True,
                environment_variables={},
            )
        return self._build_plan(
            manifest=manifest,
            base_run_dir=base_run_dir,
            launch_target=launch_target,
            assessment=assessment,
            source=str(trace_path),
            max_runs=max_runs,
        )

    def _build_plan(
        self,
        manifest: ExperimentManifest,
        base_run_dir: Path,
        launch_target: LaunchTarget,
        assessment: Optional[RobustnessAssessment],
        source: str,
        max_runs: int,
    ) -> SweepPlan:
        static_checks = [scenario.to_dict() for scenario in assessment.scenarios] if assessment is not None else []

        runs: List[SweepRun] = []
        if assessment is None or not assessment.should_reject:
            variant_specs = self._variant_specs(manifest.environment)
            for spec in variant_specs[:max_runs]:
                variant_manifest = manifest
                if spec.seed_offset:
                    variant_manifest = self._with_seed(manifest, manifest.seed + spec.seed_offset)
                variant_target = self._with_variant_environment(launch_target, spec.environment_variables)
                runs.append(
                    self._build_run(
                        manifest=variant_manifest,
                        base_run_dir=base_run_dir,
                        launch_target=variant_target,
                        name_suffix=spec.suffix,
                        rationale=spec.rationale,
                        notes=spec.notes,
                    )
                )
        else:
            runs.append(
                self._build_run(
                    manifest=manifest,
                    base_run_dir=base_run_dir,
                    launch_target=launch_target,
                    name_suffix="blocked",
                    rationale="Robustness assessment rejected this candidate; fix the reward before launching sweeps.",
                    notes="Sweep is intentionally limited until the candidate passes robustness checks.",
                )
            )

        notes = None
        if assessment is not None:
            notes = assessment.summary
        elif manifest.reward_candidate is None:
            notes = "No reward candidate attached; sweep plan only covers the baseline run."

        return SweepPlan(
            source=source,
            environment=manifest.environment,
            base_run_dir=base_run_dir,
            assessment=assessment.to_dict() if assessment is not None else None,
            static_checks=static_checks,
            runs=runs,
            notes=notes,
        )

    def _build_run(
        self,
        manifest: ExperimentManifest,
        base_run_dir: Path,
        launch_target: LaunchTarget,
        name_suffix: str,
        rationale: str,
        notes: Optional[str] = None,
    ) -> SweepRun:
        run_manifest = self._copy_manifest(
            manifest,
            suffix=name_suffix,
            baseline=launch_target.environment_variables.get("MARE_BASELINE_OVERRIDE", manifest.baseline),
        )
        run_dir = base_run_dir / run_manifest.name
        contract = self._runner().build_ppo_run_contract(run_manifest, run_dir, launch_target)
        return SweepRun(
            name=run_manifest.name,
            run_dir=run_dir,
            manifest=run_manifest,
            launch_target=launch_target,
            command=contract.render_command(),
            rationale=rationale,
            notes=notes,
        )

    def _with_seed(self, manifest: ExperimentManifest, seed: int) -> ExperimentManifest:
        return ExperimentManifest(
            name=manifest.name,
            environment=manifest.environment,
            baseline=manifest.baseline,
            seed=seed,
            created_at=manifest.created_at,
            notes=manifest.notes,
            extra=dict(manifest.extra),
            reward_candidate=manifest.reward_candidate,
        )

    def _copy_manifest(self, manifest: ExperimentManifest, suffix: str, baseline: Optional[str] = None) -> ExperimentManifest:
        return ExperimentManifest(
            name=f"{manifest.name}_{suffix}",
            environment=manifest.environment,
            baseline=baseline or manifest.baseline,
            seed=manifest.seed,
            created_at=manifest.created_at,
            notes=manifest.notes,
            extra=dict(manifest.extra),
            reward_candidate=manifest.reward_candidate,
        )

    def _with_variant_environment(
        self,
        launch_target: LaunchTarget,
        environment_variables: Dict[str, str],
    ) -> LaunchTarget:
        env = dict(launch_target.environment_variables)
        env.update(environment_variables)
        return replace(launch_target, environment_variables=env)

    def _variant_specs(self, environment: str) -> List[SweepVariantSpec]:
        if environment == "CartPole":
            return [
                SweepVariantSpec(
                    suffix="base",
                    rationale="Baseline run for the current candidate.",
                ),
                SweepVariantSpec(
                    suffix="a2c_probe",
                    rationale="Cross-check the reward under A2C to detect PPO-specific overfitting.",
                    baseline_override="A2C",
                    environment_variables={"MARE_BASELINE_OVERRIDE": "A2C"},
                    notes="Algorithm robustness probe.",
                ),
                SweepVariantSpec(
                    suffix="ddqn_probe",
                    rationale="Cross-check the reward under DDQN to detect algorithm-specific reward hacking.",
                    baseline_override="DDQN",
                    environment_variables={"MARE_BASELINE_OVERRIDE": "DDQN"},
                    notes="Algorithm robustness probe.",
                ),
                SweepVariantSpec(
                    suffix="seed_shift",
                    rationale="Nearby seed to estimate robustness across initialization noise.",
                    seed_offset=1,
                ),
                SweepVariantSpec(
                    suffix="hyperparam_probe",
                    rationale="Small learning-rate and clip-range perturbation to expose sensitivity.",
                    environment_variables={
                        "MARE_LR_SCALE": "0.5",
                        "MARE_CLIP_RANGE_SCALE": "1.2",
                    },
                ),
            ]
        if environment == "Humanoid":
            return [
                SweepVariantSpec(
                    suffix="base",
                    rationale="Baseline Humanoid run for the current candidate.",
                ),
                SweepVariantSpec(
                    suffix="a2c_probe",
                    rationale="Check whether the reward transfers from PPO to A2C on locomotion.",
                    baseline_override="A2C",
                    environment_variables={"MARE_BASELINE_OVERRIDE": "A2C"},
                    notes="Cross-algorithm transfer probe.",
                ),
                SweepVariantSpec(
                    suffix="stability_probe",
                    rationale="Seed shift plus a light entropy bias to expose fragile locomotion rewards.",
                    seed_offset=2,
                    environment_variables={
                        "MARE_ENTROPY_SCALE": "1.1",
                        "MARE_LR_SCALE": "0.75",
                    },
                    notes="Humanoid variants stay conservative to fit the 24h compute cap.",
                ),
                SweepVariantSpec(
                    suffix="clip_probe",
                    rationale="Clip-range perturbation to surface reward sensitivity in long-horizon control.",
                    environment_variables={
                        "MARE_CLIP_RANGE_SCALE": "0.9",
                        "MARE_VALUE_CLIP_SCALE": "1.1",
                    },
                ),
            ]
        if environment == "AllegroHand":
            return [
                SweepVariantSpec(
                    suffix="base",
                    rationale="Baseline AllegroHand run for the current candidate.",
                ),
                SweepVariantSpec(
                    suffix="a2c_probe",
                    rationale="Check whether the reward remains learnable under A2C.",
                    baseline_override="A2C",
                    environment_variables={"MARE_BASELINE_OVERRIDE": "A2C"},
                    notes="Cross-algorithm transfer probe.",
                ),
                SweepVariantSpec(
                    suffix="action_noise_probe",
                    rationale="Action-noise and learning-rate perturbation to expose brittle manipulation rewards.",
                    seed_offset=1,
                    environment_variables={
                        "MARE_ACTION_NOISE_SCALE": "0.8",
                        "MARE_LR_SCALE": "0.75",
                    },
                    notes="AllegroHand sweep variants should remain small and cheap.",
                ),
                SweepVariantSpec(
                    suffix="reward_clip_probe",
                    rationale="Small reward clipping adjustment to expose saturation or hacking behavior.",
                    environment_variables={
                        "MARE_REWARD_CLIP_SCALE": "0.9",
                        "MARE_CLIP_RANGE_SCALE": "1.1",
                    },
                ),
            ]
        return [
            SweepVariantSpec(
                suffix="base",
                rationale="Baseline run for the current candidate.",
            ),
            SweepVariantSpec(
                suffix="a2c_probe",
                rationale="Generic cross-algorithm probe using A2C.",
                baseline_override="A2C",
                environment_variables={"MARE_BASELINE_OVERRIDE": "A2C"},
            ),
        ]

    def _assess_candidate(
        self,
        reward_candidate_path: Optional[Path],
        entrypoint: str,
    ) -> Optional[RobustnessAssessment]:
        if reward_candidate_path is None:
            return None
        return self.analyzer.assess_file(reward_candidate_path, entrypoint=entrypoint)

    def _runner(self) -> ExperimentRunner:
        if self.runner is None:
            raise ValueError("SweepPlanner requires an ExperimentRunner")
        return self.runner

    def _trace_project_root(self, trace_path: Path) -> Path:
        parents = trace_path.parents
        if len(parents) >= 3:
            return parents[2]
        if len(parents) >= 2:
            return parents[1]
        return trace_path.parent
