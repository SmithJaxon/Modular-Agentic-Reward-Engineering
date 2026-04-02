from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.execution import PPORunDispatcher
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.manifest import ExperimentManifest
from mare.paths import ProjectPaths


def test_write_launch_script_creates_shell_script(tmp_path: Path) -> None:
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
    contract = runner.build_ppo_run_contract(manifest, tmp_path, launch_target)
    receipt = PPORunDispatcher().render_script(contract, tmp_path / "launch.sh")
    assert receipt.status == "written"
    assert (tmp_path / "launch.sh").exists()
    content = (tmp_path / "launch.sh").read_text(encoding="utf-8")
    assert "scripts/train_ppo.py" in content
    assert '. ".env"' in content
