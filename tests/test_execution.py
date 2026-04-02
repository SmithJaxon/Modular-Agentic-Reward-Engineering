from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.execution import PPORunDispatcher
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.manifest import ExperimentManifest
from mare.paths import ProjectPaths


def test_preview_receipt_contains_command() -> None:
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
    receipt = PPORunDispatcher().preview(contract)
    assert receipt.mode == "preview"
    assert receipt.status == "planned"
    assert "scripts/train_ppo.py" in receipt.command

