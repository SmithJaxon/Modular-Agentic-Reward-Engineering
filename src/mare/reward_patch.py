from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import difflib
import json

from .reward_candidate import RewardCandidateValidator


@dataclass(frozen=True)
class RewardPatchContext:
    """Trace-derived context for a reward patch recommendation."""

    trace_path: Optional[Path]
    reward_candidate_path: Path
    reward_entrypoint: str
    latest_step: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "reward_candidate_path": str(self.reward_candidate_path),
            "reward_entrypoint": self.reward_entrypoint,
            "latest_step": self.latest_step,
        }


@dataclass(frozen=True)
class RewardPatchRecommendation:
    """Heuristic patch recommendation for a reward candidate."""

    path: Path
    summary: str
    diff: str
    valid: bool
    trace_path: Optional[Path] = None
    latest_step: Optional[Dict[str, Any]] = None
    trace_context: Optional[RewardPatchContext] = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "summary": self.summary,
            "diff": self.diff,
            "valid": self.valid,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "latest_step": self.latest_step,
            "trace_context": self.trace_context.to_dict() if self.trace_context is not None else None,
        }


class RewardPatchRecommender:
    """Generate a small unified diff for a reward candidate."""

    def __init__(self, validator: RewardCandidateValidator | None = None) -> None:
        self.validator = validator or RewardCandidateValidator()

    def recommend(self, path: Path, entrypoint: str = "compute_reward") -> RewardPatchRecommendation:
        return self._recommend(path=path, entrypoint=entrypoint, trace_path=None, latest_step=None)

    def recommend_from_trace(self, trace_path: Path) -> RewardPatchRecommendation:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest = trace_data.get("manifest", {})
        reward_candidate = manifest.get("reward_candidate") or trace_data.get("reward_candidate") or {}
        candidate_path = Path(reward_candidate.get("path", ""))
        entrypoint = str(reward_candidate.get("entrypoint", "compute_reward"))
        latest_step = self._latest_step(trace_data)
        context = RewardPatchContext(
            trace_path=trace_path,
            reward_candidate_path=self._resolve_candidate_path(candidate_path, trace_path),
            reward_entrypoint=entrypoint,
            latest_step=latest_step,
        )
        return self._recommend(
            path=context.reward_candidate_path,
            entrypoint=context.reward_entrypoint,
            trace_path=context.trace_path,
            latest_step=context.latest_step,
            trace_context=context,
        )

    def _recommend(
        self,
        path: Path,
        entrypoint: str,
        trace_path: Optional[Path],
        latest_step: Optional[Dict[str, Any]],
        trace_context: Optional[RewardPatchContext] = None,
    ) -> RewardPatchRecommendation:
        source = path.read_text(encoding="utf-8")
        validation = self.validator.validate_file(path, entrypoint=entrypoint)
        lines = source.splitlines(keepends=True)
        modified_lines = self._apply_shape_patch(lines, latest_step=latest_step)
        diff = "".join(
            difflib.unified_diff(
                lines,
                modified_lines,
                fromfile=str(path),
                tofile=str(path),
                lineterm="",
            )
        )
        summary = self._build_summary(validation.valid, latest_step)
        return RewardPatchRecommendation(
            path=path,
            summary=summary,
            diff=diff,
            valid=validation.valid,
            trace_path=trace_path,
            latest_step=latest_step,
            trace_context=trace_context,
        )

    def _build_summary(self, valid: bool, latest_step: Optional[Dict[str, Any]]) -> str:
        if not valid:
            return (
                "Fix validation issues before reward shaping; the patch recommendation is based on the latest "
                "orchestration trace."
            )

        if latest_step is None:
            return "Add a small shaping term to the reward return expression to reduce sparse-reward behavior."

        decision = str(latest_step.get("decision", {}).get("decision", ""))
        tool_name = str(latest_step.get("result", {}).get("tool_name", ""))
        environment = str(
            latest_step.get("result", {})
            .get("payload", {})
            .get("contract", {})
            .get("execution_plan", {})
            .get("environment", "")
        )
        if decision == "preview_launch" and environment:
            return (
                "Latest orchestration step previewed {0}; add a small shaping term tied to the state features "
                "before re-running the preview."
            ).format(environment)
        if tool_name == "validate_reward_candidate":
            return "Validation passed; use the latest trace to add a small shaping term before launch."
        return "Use the latest orchestration trace to add a small shaping term that reduces sparse-reward behavior."

    def _apply_shape_patch(
        self,
        lines: List[str],
        latest_step: Optional[Dict[str, Any]],
    ) -> List[str]:
        modified = list(lines)
        target_index = self._find_return_line(modified)
        if target_index is None:
            return modified

        return_line = modified[target_index]
        latest_environment = self._latest_environment(latest_step)
        if latest_environment == "CartPole" or ("position" in "".join(modified) and "velocity" in "".join(modified)):
            shaping = " + 0.02 * (1.0 - min(abs(position), 1.0))"
        elif latest_environment == "Humanoid":
            shaping = " + 0.01 * (1.0 - min(abs(observation[0]), 1.0))"
        elif latest_environment == "AllegroHand":
            shaping = " + 0.01 * (1.0 - min(abs(observation[0]), 1.0))"
        else:
            shaping = " + 0.02 * (1.0 - min(abs(observation[0]), 1.0))"

        if shaping.strip() in return_line:
            return modified

        stripped = return_line.rstrip("\n")
        newline = "\n" if return_line.endswith("\n") else ""
        modified[target_index] = f"{stripped}{shaping}{newline}"
        return modified

    def _find_return_line(self, lines: List[str]) -> int | None:
        in_target_function = False
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def compute_reward"):
                in_target_function = True
                continue
            if in_target_function:
                if stripped.startswith("def ") and not stripped.startswith("def compute_reward"):
                    return None
                if stripped.startswith("return "):
                    return index
        return None

    def _latest_step(self, trace_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        steps = trace_data.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return None
        latest = steps[-1]
        return latest if isinstance(latest, dict) else None

    def _latest_environment(self, latest_step: Optional[Dict[str, Any]]) -> Optional[str]:
        if latest_step is None:
            return None
        payload = latest_step.get("result", {}).get("payload", {})
        contract = payload.get("contract", {})
        execution_plan = contract.get("execution_plan", {})
        environment = execution_plan.get("environment")
        return str(environment) if environment else None

    def _resolve_candidate_path(self, candidate_path: Path, trace_path: Path) -> Path:
        if candidate_path.is_absolute():
            return candidate_path
        trace_relative = (trace_path.parent / candidate_path).resolve()
        if trace_relative.exists():
            return trace_relative
        return candidate_path.resolve()
