from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RewardDraft:
    """A generated or refined reward candidate on disk."""

    path: Path
    summary: str
    entrypoint: str = "compute_reward"


class RewardDesignStudio:
    """Small local helper for drafting and refining reward candidates."""

    def draft(self, environment: str, output_path: Path, human_feedback: Optional[str] = None) -> RewardDraft:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = self._template_for(environment)
        if human_feedback:
            source = self._inject_feedback_comment(source, human_feedback)
        output_path.write_text(source, encoding="utf-8")
        summary = "Drafted a starter reward candidate for {0}.".format(environment)
        if human_feedback:
            summary += " Included the latest human feedback as guidance comments."
        return RewardDraft(path=output_path, summary=summary)

    def refine(
        self,
        source_path: Path,
        output_path: Path,
        summary_hint: str,
        human_feedback: Optional[str] = None,
    ) -> RewardDraft:
        source = source_path.read_text(encoding="utf-8")
        refined = self._refine_source(source, human_feedback=human_feedback)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(refined, encoding="utf-8")
        summary = summary_hint
        if human_feedback:
            summary += " Incorporated human feedback guidance."
        return RewardDraft(path=output_path, summary=summary)

    def _template_for(self, environment: str) -> str:
        if environment == "CartPole":
            body = [
                '"""Auto-generated starter reward for CartPole."""',
                "",
                "def compute_reward(observation, action, info):",
                "    position = float(observation[0])",
                "    velocity = float(observation[1])",
                "    clipped_position = min(abs(position), 2.4)",
                "    clipped_velocity = min(abs(velocity), 3.0)",
                "    center_bonus = 1.0 - (clipped_position / 2.4)",
                "    stability_bonus = 1.0 - (clipped_velocity / 3.0)",
                "    action_penalty = 0.01 * abs(float(action))",
                "    done_penalty = 1.0 if info.get('terminated') else 0.0",
                "    return center_bonus + 0.25 * stability_bonus - action_penalty - done_penalty",
            ]
            return "\n".join(body) + "\n"
        if environment == "Humanoid":
            body = [
                '"""Auto-generated starter reward for Humanoid."""',
                "",
                "def compute_reward(observation, action, info):",
                "    torso_signal = 1.0 - min(abs(float(observation[0])), 1.0)",
                "    action_penalty = 0.001 * sum(abs(float(value)) for value in action[: min(len(action), 8)])",
                "    fall_penalty = 1.0 if info.get('terminated') else 0.0",
                "    return torso_signal - action_penalty - fall_penalty",
            ]
            return "\n".join(body) + "\n"
        if environment == "AllegroHand":
            body = [
                '"""Auto-generated starter reward for Allegro Hand."""',
                "",
                "def compute_reward(observation, action, info):",
                "    pose_signal = 1.0 - min(abs(float(observation[0])), 1.0)",
                "    action_penalty = 0.001 * sum(abs(float(value)) for value in action[: min(len(action), 8)])",
                "    drop_penalty = 1.0 if info.get('terminated') else 0.0",
                "    return pose_signal - action_penalty - drop_penalty",
            ]
            return "\n".join(body) + "\n"
        body = [
            '"""Auto-generated generic reward candidate."""',
            "",
            "def compute_reward(observation, action, info):",
            "    primary_signal = 1.0 - min(abs(float(observation[0])), 1.0)",
            "    action_penalty = 0.001 * abs(float(action if not isinstance(action, (list, tuple)) else action[0]))",
            "    return primary_signal - action_penalty",
        ]
        return "\n".join(body) + "\n"

    def _inject_feedback_comment(self, source: str, human_feedback: str) -> str:
        lines = source.splitlines()
        comment = "# Human feedback: {0}".format(" ".join(human_feedback.strip().split()))
        insert_at = 1 if lines and lines[0].startswith('"""') else 0
        lines.insert(insert_at, comment)
        return "\n".join(lines) + "\n"

    def _refine_source(self, source: str, human_feedback: Optional[str]) -> str:
        lines = source.splitlines()
        updated_lines = list(lines)
        if human_feedback:
            feedback_comment = "# Human feedback: {0}".format(" ".join(human_feedback.strip().split()))
            if feedback_comment not in updated_lines:
                insert_at = 1 if updated_lines and updated_lines[0].startswith('"""') else 0
                updated_lines.insert(insert_at, feedback_comment)

        for index, line in enumerate(updated_lines):
            stripped = line.lstrip()
            if not stripped.startswith("return "):
                continue
            if "min(" in stripped or "max(" in stripped:
                return "\n".join(updated_lines) + "\n"
            indent = line[: len(line) - len(stripped)]
            expression = stripped[len("return "):]
            updated_lines[index] = "{0}return max(-5.0, min(5.0, {1}))".format(indent, expression)
            return "\n".join(updated_lines) + "\n"
        return "\n".join(updated_lines) + "\n"
