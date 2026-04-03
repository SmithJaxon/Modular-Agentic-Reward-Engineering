from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.orchestration import AgenticOrchestrator
from mare.paths import ProjectPaths
from mare.sweep import SweepPlanner


def test_sweep_planner_builds_variants_from_manifest(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=Path.cwd(),
    )

    plan = SweepPlanner(runner=runner).build_from_manifest(
        manifest=manifest,
        base_run_dir=tmp_path,
        launch_target=launch_target,
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
        max_runs=3,
    )

    assert plan.environment == "CartPole"
    assert plan.assessment is not None
    assert len(plan.static_checks) == 3
    assert len(plan.runs) >= 2
    assert plan.runs[0].command[0] == "python3"
    assert plan.runs[0].run_dir.name.endswith("_base")
    assert not (tmp_path / "sweep_plan.json").exists()


def test_sweep_planner_uses_environment_specific_variants(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=Path.cwd(),
    )

    humanoid_manifest = _create_manifest(Path("configs/presets/humanoid_ppo.yaml"))
    humanoid_plan = SweepPlanner(runner=runner).build_from_manifest(
        manifest=humanoid_manifest,
        base_run_dir=tmp_path / "humanoid",
        launch_target=launch_target,
        max_runs=4,
    )
    humanoid_names = [run.name for run in humanoid_plan.runs]
    assert humanoid_names == [
        "humanoid_ppo_baseline_base",
        "humanoid_ppo_baseline_a2c_probe",
        "humanoid_ppo_baseline_stability_probe",
        "humanoid_ppo_baseline_clip_probe",
    ]
    assert humanoid_plan.runs[1].manifest.baseline == "A2C"
    assert humanoid_plan.runs[2].launch_target.environment_variables["MARE_ENTROPY_SCALE"] == "1.1"
    assert humanoid_plan.runs[3].launch_target.environment_variables["MARE_VALUE_CLIP_SCALE"] == "1.1"

    allegro_manifest = _create_manifest(Path("configs/presets/allegro_hand_ppo.yaml"))
    allegro_plan = SweepPlanner(runner=runner).build_from_manifest(
        manifest=allegro_manifest,
        base_run_dir=tmp_path / "allegro",
        launch_target=launch_target,
        max_runs=4,
    )
    allegro_names = [run.name for run in allegro_plan.runs]
    assert allegro_names == [
        "allegro_hand_ppo_baseline_base",
        "allegro_hand_ppo_baseline_a2c_probe",
        "allegro_hand_ppo_baseline_action_noise_probe",
        "allegro_hand_ppo_baseline_reward_clip_probe",
    ]
    assert allegro_plan.runs[1].manifest.baseline == "A2C"
    assert allegro_plan.runs[2].launch_target.environment_variables["MARE_ACTION_NOISE_SCALE"] == "0.8"
    assert allegro_plan.runs[3].launch_target.environment_variables["MARE_REWARD_CLIP_SCALE"] == "0.9"


def test_sweep_planner_builds_from_trace_and_writes_plan(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=Path.cwd(),
    )
    trace = AgenticOrchestrator(runner=runner).run(
        manifest=manifest,
        run_dir=tmp_path,
        launch_target=launch_target,
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
    )
    trace_path = trace.write(tmp_path / "orchestration.json")

    plan = SweepPlanner(runner=runner).build_from_trace(trace_path, base_run_dir=tmp_path)
    plan_path = plan.write(tmp_path / "sweep_plan.json")

    assert plan.source == str(trace_path)
    assert plan_path.exists()
    assert plan.runs
    assert plan.notes


def test_cli_sweep_plan_accepts_trace(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=Path.cwd(),
    )
    trace = AgenticOrchestrator(runner=runner).run(
        manifest=manifest,
        run_dir=tmp_path,
        launch_target=launch_target,
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
    )
    trace_path = trace.write(tmp_path / "orchestration.json")

    exit_code = main(["sweep-plan", "--trace", str(trace_path), "--max-runs", "2"])
    assert exit_code == 0
