from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import main
from mare.registry import list_baseline_presets, validate_registry
from mare.runtime import load_experiment_spec
from mare.schema import validate_experiment_spec_data


def test_validate_experiment_spec_data_accepts_cartpole() -> None:
    spec = validate_experiment_spec_data(
        {
            "name": "cartpole_ppo_baseline",
            "environment": "CartPole",
            "baseline": "PPO",
            "seed": 7,
        }
    )
    assert spec.environment == "CartPole"
    assert spec.baseline == "PPO"


def test_registry_covers_all_environments() -> None:
    validate_registry()
    environments = {preset.environment for preset in list_baseline_presets()}
    assert environments == {"CartPole", "Humanoid", "AllegroHand"}


def test_load_experiment_spec_reads_example_config() -> None:
    spec = load_experiment_spec(Path("configs/example_experiment.yaml"))
    assert spec.name == "cartpole_baseline_dry_run"
    assert spec.environment == "CartPole"


def test_cli_dry_run_returns_success() -> None:
    exit_code = main(["dry-run", "--config", "configs/example_experiment.yaml"])
    assert exit_code == 0

