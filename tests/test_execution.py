from __future__ import annotations

from dataclasses import replace
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


def test_local_dispatch_executes_cartpole_training(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    manifest = ExperimentManifest(
        name="cartpole_local_dispatch",
        environment="CartPole",
        baseline="PPO",
        seed=3,
    )
    launch_target = LaunchTarget(
        kind="local",
        python_executable="python3",
        working_directory=Path.cwd(),
        gpu_required=False,
    )
    contract = runner.build_ppo_run_contract(manifest, tmp_path, launch_target)
    contract = replace(
        contract,
        execution_plan=replace(contract.execution_plan, train_steps=64, eval_episodes=2, device="cpu"),
    )
    receipt = PPORunDispatcher().dispatch(contract)
    assert receipt.mode == "dispatch"
    assert receipt.status == "completed"
    assert (tmp_path / "result.json").exists()


def test_local_dispatch_falls_back_when_cuda_is_unusable(tmp_path: Path) -> None:
    runner = ExperimentRunner(ProjectPaths(Path.cwd()))
    manifest = ExperimentManifest(
        name="cartpole_local_cuda_fallback",
        environment="CartPole",
        baseline="PPO",
        seed=4,
    )
    launch_target = LaunchTarget(
        kind="local",
        python_executable="python3",
        working_directory=Path.cwd(),
        gpu_required=False,
    )
    contract = runner.build_ppo_run_contract(manifest, tmp_path, launch_target)
    contract = replace(
        contract,
        execution_plan=replace(contract.execution_plan, train_steps=32, eval_episodes=1, device="cuda"),
    )
    receipt = PPORunDispatcher().dispatch(contract)
    assert receipt.status == "completed"
    assert (tmp_path / "result.json").exists()
    result_text = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in result_text
