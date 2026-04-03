from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.adapters import get_environment_adapter
from mare.environment_registry import list_environment_presets
from mare.manifest import ExperimentManifest


def test_environment_registry_contains_three_targets() -> None:
    names = [preset.profile.name for preset in list_environment_presets()]
    assert names == ["AllegroHand", "CartPole", "Humanoid"]


def test_environment_adapter_builds_execution_plan() -> None:
    adapter = get_environment_adapter("CartPole")
    manifest = ExperimentManifest(
        name="cartpole_ppo_baseline",
        environment="CartPole",
        baseline="PPO",
        seed=7,
    )
    report = adapter.evaluate(manifest)
    assert report.status == "evaluation_planned"
    assert report.metrics["train_steps"] == 50000.0
    assert report.metrics["eval_episodes"] == 10.0


def test_environment_device_defaults_match_runtime_policy() -> None:
    cartpole = get_environment_adapter("CartPole").build_training_plan(
        ExperimentManifest(name="cartpole", environment="CartPole", baseline="PPO", seed=1)
    )
    humanoid = get_environment_adapter("Humanoid").build_training_plan(
        ExperimentManifest(name="humanoid", environment="Humanoid", baseline="PPO", seed=1)
    )
    allegro = get_environment_adapter("AllegroHand").build_training_plan(
        ExperimentManifest(name="allegro", environment="AllegroHand", baseline="PPO", seed=1)
    )
    assert cartpole.device == "auto"
    assert humanoid.device == "cuda"
    assert allegro.device == "cuda"
