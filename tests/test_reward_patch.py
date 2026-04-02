from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.orchestration import AgenticOrchestrator
from mare.paths import ProjectPaths
from mare.reward_patch import RewardPatchRecommender


def test_reward_patch_recommender_uses_orchestration_trace(tmp_path: Path) -> None:
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

    recommendation = RewardPatchRecommender().recommend_from_trace(trace_path)

    assert recommendation.trace_path == trace_path
    assert recommendation.latest_step is not None
    assert "latest orchestration trace" in recommendation.summary
    assert recommendation.diff
    assert recommendation.trace_context is not None
    assert recommendation.trace_context.reward_entrypoint == "compute_reward"
    assert recommendation.trace_context.reward_candidate_path.name == "cartpole_reward.py"


def test_cli_recommend_reward_patch_accepts_trace(tmp_path: Path) -> None:
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

    exit_code = main(["recommend-reward-patch", "--trace", str(trace_path)])
    assert exit_code == 0
