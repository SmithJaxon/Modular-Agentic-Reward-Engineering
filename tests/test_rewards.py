from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mare.reward_candidate import RewardCandidateLoader, RewardCandidateValidator


def test_reward_candidate_validator_accepts_example_candidate() -> None:
    path = Path("reward_candidates/cartpole_reward.py")
    result = RewardCandidateValidator().validate_file(path)
    assert result.valid
    assert result.issues == []


def test_reward_candidate_loader_executes_example_candidate() -> None:
    candidate = RewardCandidateLoader().load(Path("reward_candidates/cartpole_reward.py"))
    reward = candidate.reward([0.0, 0.0], 0, {})
    assert isinstance(reward, float)


def test_reward_candidate_loader_rejects_missing_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "missing_entrypoint.py"
    path.write_text("x = 1\n", encoding="utf-8")
    result = RewardCandidateValidator().validate_file(path)
    assert not result.valid
    assert any("Missing required function" in issue.message for issue in result.issues)


def test_reward_candidate_validator_rejects_imports(tmp_path: Path) -> None:
    path = tmp_path / "bad_reward.py"
    path.write_text(
        "import os\n"
        "def compute_reward(observation, action, info):\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    result = RewardCandidateValidator().validate_file(path)
    assert not result.valid
    assert any("Import not allowed" in issue.message for issue in result.issues)
