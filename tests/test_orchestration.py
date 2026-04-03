from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.cli import _create_manifest, _load_reward_candidate_metadata, main
from mare.experiment import ExperimentRunner
from mare.launch import LaunchTarget
from mare.orchestration import (
    AgenticOrchestrator,
    HeuristicOrchestrationPolicy,
    OpenAIOrchestrationPolicy,
    OrchestrationState,
)
from mare.paths import ProjectPaths


def test_orchestrator_runs_validate_preview_and_script(tmp_path: Path) -> None:
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
    assert trace.status == "completed"
    assert [step.decision.decision for step in trace.steps] == [
        "validate_reward_candidate",
        "assess_reward_robustness",
        "preview_launch",
        "write_launch_script",
    ]
    assert trace.steps[0].decision.rationale
    assert trace.steps[0].decision.suggested_patch
    trace_path = trace.write(tmp_path / "orchestration.json")
    assert trace_path.exists()
    assert (tmp_path / "launch.sh").exists()


def test_cli_orchestrate_returns_success() -> None:
    exit_code = main(
        [
            "orchestrate",
            "--config",
            "configs/example_experiment.yaml",
            "--reward-candidate",
            "reward_candidates/cartpole_reward.py",
        ]
    )
    assert exit_code == 0


def test_openai_policy_uses_fake_responses_client() -> None:
    class FakeResponse:
        output_text = '{"decision":"preview_launch","rationale":"continue","suggested_patch":"Add a tiny shaping term tied to a key state feature.","confidence":0.9}'

    class FakeResponses:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    state = OrchestrationState(
        manifest=manifest,
        run_dir=Path("runs/example"),
        launch_target=LaunchTarget(
            kind="gpu_vm",
            python_executable="python3",
            working_directory=Path.cwd(),
        ),
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
    )
    policy = OpenAIOrchestrationPolicy(client=FakeClient(), fallback=HeuristicOrchestrationPolicy())
    decision = policy.decide(state)
    assert decision.decision == "preview_launch"
    assert decision.rationale == "continue"
    assert decision.suggested_patch.startswith("Add a tiny shaping term")
    assert decision.confidence == 0.9


def test_heuristic_policy_records_rationale() -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
    state = OrchestrationState(
        manifest=manifest,
        run_dir=Path("runs/example"),
        launch_target=LaunchTarget(
            kind="gpu_vm",
            python_executable="python3",
            working_directory=Path.cwd(),
        ),
        reward_candidate_path=Path("reward_candidates/cartpole_reward.py"),
    )
    decision = HeuristicOrchestrationPolicy().decide(state)
    assert decision.decision == "validate_reward_candidate"
    assert decision.rationale
    assert decision.suggested_patch


def test_orchestrator_can_generate_reward_candidate_when_missing(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
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
    )
    decisions = [step.decision.decision for step in trace.steps]
    assert decisions[0] == "generate_reward_candidate"
    assert "validate_reward_candidate" in decisions
    assert (tmp_path / "generated_reward.py").exists()
    assert trace.reward_candidate is not None


def test_orchestrator_refines_generated_candidate_when_feedback_present(tmp_path: Path) -> None:
    manifest = _create_manifest(Path("configs/example_experiment.yaml"))
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
        human_feedback="Prefer bounded rewards and penalize termination strongly.",
    )
    decisions = [step.decision.decision for step in trace.steps]
    assert "generate_reward_candidate" in decisions
    assert "refine_reward_candidate" in decisions
    assert (tmp_path / "refined_reward.py").exists()
