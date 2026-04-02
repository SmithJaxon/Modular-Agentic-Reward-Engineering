from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.orchestration import AgenticOrchestrator
from mare.paths import ProjectPaths
from mare.report import ProjectReporter


def _write_completed_run(runner: ExperimentRunner, run_dir: Path, seed: int) -> Path:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.seed = seed
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    runner.save_manifest(run_dir, manifest)
    report = runner.placeholder_report(manifest, run_dir)
    report.metrics["evaluation_score"] = 0.5 + float(seed) / 100.0
    runner.save_result(run_dir, report)
    return run_dir


def test_project_report_collects_runs_and_trace(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(tmp_path))
    runs_root = tmp_path / "runs"
    baseline_dir = _write_completed_run(runner, runs_root / "baseline", seed=7)
    candidate_dir = _write_completed_run(runner, runs_root / "candidate", seed=8)

    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=tmp_path,
    )
    trace = AgenticOrchestrator(runner=runner).run(
        manifest=manifest,
        run_dir=baseline_dir,
        launch_target=launch_target,
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
    )
    trace.write(baseline_dir / "orchestration.json")

    report = ProjectReporter().build_report(project_root=tmp_path)

    assert report.run_count == 2
    assert report.trace_count == 1
    assert report.latest_trace_path == baseline_dir / "orchestration.json"
    assert report.readiness_status is not None
    assert report.benchmark_aggregate is not None
    assert len(report.run_snapshots) == 2


def test_cli_project_report_writes_artifact(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(tmp_path))
    runs_root = tmp_path / "runs"
    _write_completed_run(runner, runs_root / "baseline", seed=7)
    _write_completed_run(runner, runs_root / "candidate", seed=8)

    exit_code = main(["project-report", "--project-root", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "artifacts" / "project_report.json").exists()


def test_cli_project_brief_accepts_project_root(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(tmp_path))
    runs_root = tmp_path / "runs"
    _write_completed_run(runner, runs_root / "baseline", seed=7)
    _write_completed_run(runner, runs_root / "candidate", seed=8)

    exit_code = main(["project-brief", "--project-root", str(tmp_path)])
    assert exit_code == 0
