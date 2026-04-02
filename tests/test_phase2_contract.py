from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.manifest import ExperimentManifest
from mare.paths import ProjectPaths


def test_ppo_run_contract_renders_command() -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    manifest = ExperimentManifest(
        name="cartpole_ppo_baseline",
        environment="CartPole",
        baseline="PPO",
        seed=7,
    )
    launch_target = LaunchTarget(
        kind="gpu_vm",
        python_executable="python3",
        working_directory=Path.cwd(),
    )
    contract = runner.build_ppo_run_contract(manifest, Path("runs/cartpole"), launch_target)
    command = contract.render_command()
    assert command[0] == "python3"
    assert "--environment" in command
    assert "CartPole" in command


def test_cli_plan_returns_success() -> None:
    exit_code = main(["plan", "--config", "configs/example_experiment.yaml"])
    assert exit_code == 0

