from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest
from mare.experiment import ExperimentRunner
from mare.contracts import RunReport
from mare.paths import ProjectPaths
from mare.reward_candidate import RewardCandidateLoader


def test_manifest_persists_reward_candidate_metadata(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    candidate = RewardCandidateLoader().load(Path("reward_candidates/cartpole_reward.py"))
    manifest.reward_candidate = {
        "name": candidate.spec.name,
        "path": str(candidate.spec.path),
        "entrypoint": candidate.spec.entrypoint,
    }
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = runner.save_manifest(run_dir, manifest)
    payload = manifest_path.read_text(encoding="utf-8")
    assert '"reward_candidate"' in payload
    assert '"cartpole_reward"' in payload


def test_run_report_serializes_reward_candidate_metadata(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    manifest.reward_candidate = {
        "name": "cartpole_reward",
        "path": "reward_candidates/cartpole_reward.py",
        "entrypoint": "compute_reward",
    }
    report = RunReport(
        manifest=manifest,
        status="evaluation_planned",
        reward_candidate=manifest.reward_candidate,
    )
    payload = report.to_dict()
    assert payload["reward_candidate"]["name"] == "cartpole_reward"
