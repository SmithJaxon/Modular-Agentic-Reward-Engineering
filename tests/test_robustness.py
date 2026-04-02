from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.orchestration import AgenticOrchestrator
from mare.paths import ProjectPaths
from mare.robustness import RewardRobustnessAnalyzer
from mare.sweep import SweepPlanner


def test_reward_robustness_analyzer_scores_example_candidate() -> None:
    assessment = RewardRobustnessAnalyzer().assess_file(Path("reward_candidates/cartpole_reward.py"))

    assert assessment.valid
    assert assessment.environment == "CartPole"
    assert assessment.overall_score > 0.5
    assert assessment.risk_level in {"low", "medium", "high", "critical"}
    assert assessment.scenarios
    assert assessment.signals


def test_reward_robustness_analyzer_uses_trace_context(tmp_path: Path) -> None:
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

    assessment = RewardRobustnessAnalyzer().assess_trace(trace_path)

    assert assessment.trace_path == trace_path
    assert assessment.latest_step is not None
    assert assessment.summary
    assert assessment.scenarios


def test_cli_reward_robustness_accepts_trace(tmp_path: Path) -> None:
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

    exit_code = main(["reward-robustness", "--trace", str(trace_path)])
    assert exit_code == 0


def test_run_summary_combines_orchestration_and_robustness(tmp_path: Path) -> None:
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

    summary = RewardRobustnessAnalyzer().summarize_trace(trace_path)

    assert summary.orchestration_status == "completed"
    assert summary.step_count == 4
    assert summary.final_decision == "write_launch_script"
    assert summary.robustness_assessment is not None


def test_cli_run_summary_accepts_trace(tmp_path: Path) -> None:
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

    exit_code = main(["run-summary", "--trace", str(trace_path)])
    assert exit_code == 0


def test_compare_report_combines_trace_and_sweep(tmp_path: Path) -> None:
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
    plan.write(tmp_path / "sweep_plan.json")

    report = RewardRobustnessAnalyzer().compare_trace_with_sweep(trace_path)

    assert report.sweep_plan_path == tmp_path / "sweep_plan.json"
    assert report.sweep_plan is not None
    assert report.orchestration_vs_sweep_aligned is True
    assert "alignment=matched" in report.summary


def test_cli_compare_report_accepts_trace(tmp_path: Path) -> None:
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
    plan.write(tmp_path / "sweep_plan.json")

    exit_code = main(["compare-report", "--trace", str(trace_path)])
    assert exit_code == 0


def test_review_brief_formats_text(tmp_path: Path) -> None:
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
    SweepPlanner(runner=runner).build_from_trace(trace_path, base_run_dir=tmp_path).write(
        tmp_path / "sweep_plan.json"
    )

    brief = RewardRobustnessAnalyzer().build_review_brief(trace_path)
    text = brief.to_text()

    assert text.startswith("Phase 5 Review Brief")
    assert "Trace status: completed" in text
    assert "Sweep plan: 3 runs" in text
    assert "Recommendation:" in text


def test_cli_review_brief_accepts_trace(tmp_path: Path) -> None:
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
    SweepPlanner(runner=runner).build_from_trace(trace_path, base_run_dir=tmp_path).write(
        tmp_path / "sweep_plan.json"
    )

    exit_code = main(["review-brief", "--trace", str(trace_path)])
    assert exit_code == 0


def test_readiness_status_reports_ready(tmp_path: Path) -> None:
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
    SweepPlanner(runner=runner).build_from_trace(trace_path, base_run_dir=tmp_path).write(
        tmp_path / "sweep_plan.json"
    )

    status = RewardRobustnessAnalyzer().build_readiness_status(trace_path)

    assert status.ready
    assert status.label == "READY"
    assert "trace=completed" in status.detail


def test_cli_phase5_status_accepts_trace(tmp_path: Path) -> None:
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
    SweepPlanner(runner=runner).build_from_trace(trace_path, base_run_dir=tmp_path).write(
        tmp_path / "sweep_plan.json"
    )

    exit_code = main(["phase5-status", "--trace", str(trace_path)])
    assert exit_code == 0
