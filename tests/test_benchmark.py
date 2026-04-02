from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.benchmark import BenchmarkReporter
from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.paths import ProjectPaths


def _write_placeholder_run(runner: ExperimentRunner, run_dir: Path, seed: int) -> Path:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.seed = seed
    manifest.reward_candidate = _load_reward_candidate_metadata(
        Path("reward_candidates/cartpole_reward.py"),
        "compute_reward",
    )
    runner.save_manifest(run_dir, manifest)
    report = runner.placeholder_report(manifest, run_dir)
    report.metrics["evaluation_score"] = float(seed) / 10.0
    runner.save_result(run_dir, report)
    return run_dir


def test_benchmark_report_compares_two_runs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    baseline_dir = _write_placeholder_run(runner, tmp_path / "baseline", seed=7)
    candidate_dir = _write_placeholder_run(runner, tmp_path / "candidate", seed=8)

    report = BenchmarkReporter().compare_runs(baseline_dir, candidate_dir, metric_priority=["evaluation_score"])

    assert report.environment == "CartPole"
    assert report.comparisons
    assert report.comparisons[0].deltas[0].metric == "evaluation_score"
    assert report.comparisons[0].summary


def test_cli_benchmark_compare_accepts_run_dirs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    baseline_dir = _write_placeholder_run(runner, tmp_path / "baseline", seed=7)
    candidate_dir = _write_placeholder_run(runner, tmp_path / "candidate", seed=8)

    exit_code = main(
        [
            "benchmark-compare",
            "--baseline-run-dir",
            str(baseline_dir),
            "--candidate-run-dir",
            str(candidate_dir),
            "--metric",
            "evaluation_score",
        ]
    )
    assert exit_code == 0


def test_benchmark_brief_formats_text(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    baseline_dir = _write_placeholder_run(runner, tmp_path / "baseline", seed=7)
    candidate_dir = _write_placeholder_run(runner, tmp_path / "candidate", seed=8)

    brief = BenchmarkReporter().build_brief(baseline_dir, candidate_dir, metric_priority=["evaluation_score"])
    text = brief.to_text()

    assert text.startswith("Benchmark Brief")
    assert "Baseline run:" in text
    assert "Candidate run:" in text
    assert "evaluation_score:" in text


def test_cli_benchmark_brief_accepts_run_dirs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    baseline_dir = _write_placeholder_run(runner, tmp_path / "baseline", seed=7)
    candidate_dir = _write_placeholder_run(runner, tmp_path / "candidate", seed=8)

    exit_code = main(
        [
            "benchmark-brief",
            "--baseline-run-dir",
            str(baseline_dir),
            "--candidate-run-dir",
            str(candidate_dir),
            "--metric",
            "evaluation_score",
        ]
    )
    assert exit_code == 0


def test_benchmark_aggregate_summarizes_multiple_runs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    first_dir = _write_placeholder_run(runner, tmp_path / "first", seed=7)
    second_dir = _write_placeholder_run(runner, tmp_path / "second", seed=8)
    third_dir = _write_placeholder_run(runner, tmp_path / "third", seed=9)

    report = BenchmarkReporter().aggregate_runs([first_dir, second_dir, third_dir], metric_priority=["evaluation_score"])

    assert report.run_count == 3
    assert report.environment == "CartPole"
    assert report.metric_summary["evaluation_score"]["avg"] is not None


def test_benchmark_aggregate_brief_formats_text(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    first_dir = _write_placeholder_run(runner, tmp_path / "first", seed=7)
    second_dir = _write_placeholder_run(runner, tmp_path / "second", seed=8)

    brief = BenchmarkReporter().build_aggregate_brief([first_dir, second_dir], metric_priority=["evaluation_score"])
    text = brief.to_text()

    assert text.startswith("Benchmark Aggregate Brief")
    assert "Run count: 2" in text
    assert "evaluation_score:" in text


def test_cli_benchmark_aggregate_accepts_run_dirs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    first_dir = _write_placeholder_run(runner, tmp_path / "first", seed=7)
    second_dir = _write_placeholder_run(runner, tmp_path / "second", seed=8)

    exit_code = main(
        [
            "benchmark-aggregate",
            "--run-dir",
            str(first_dir),
            "--run-dir",
            str(second_dir),
            "--metric",
            "evaluation_score",
        ]
    )
    assert exit_code == 0


def test_cli_benchmark_aggregate_brief_accepts_run_dirs(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    first_dir = _write_placeholder_run(runner, tmp_path / "first", seed=7)
    second_dir = _write_placeholder_run(runner, tmp_path / "second", seed=8)

    exit_code = main(
        [
            "benchmark-aggregate-brief",
            "--run-dir",
            str(first_dir),
            "--run-dir",
            str(second_dir),
            "--metric",
            "evaluation_score",
        ]
    )
    assert exit_code == 0
