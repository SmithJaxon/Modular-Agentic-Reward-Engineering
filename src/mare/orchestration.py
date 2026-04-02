from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import json
import os

from .experiment import ExperimentRunner
from .launch import LaunchTarget
from .manifest import ExperimentManifest
from .paths import ProjectPaths
from .reward_candidate import RewardCandidateLoader, RewardCandidateValidator
from .robustness import RewardRobustnessAnalyzer, RobustnessAssessment


@dataclass(frozen=True)
class OrchestrationToolResult:
    """Result returned by a single orchestration tool call."""

    tool_name: str
    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestrationDecision:
    """Structured decision emitted by the orchestration policy."""

    decision: str
    rationale: str
    suggested_edit: Optional[str] = None
    suggested_patch: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestrationStep:
    """Single decision/tool step in the orchestration loop."""

    index: int
    decision: OrchestrationDecision
    result: OrchestrationToolResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "decision": self.decision.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass
class OrchestrationState:
    """Mutable loop state for candidate validation and launch planning."""

    manifest: ExperimentManifest
    run_dir: Path
    launch_target: LaunchTarget
    reward_candidate_path: Optional[Path]
    reward_entrypoint: str = "compute_reward"
    validated: bool = False
    valid: Optional[bool] = None
    robustness_checked: bool = False
    robustness_rejected: bool = False
    robustness_score: Optional[float] = None
    robustness_risk: Optional[str] = None
    previewed: bool = False
    scripted: bool = False
    current_step: int = 0
    max_steps: int = 4


@dataclass(frozen=True)
class OrchestrationTrace:
    """Serialized trace of an orchestration run."""

    manifest: ExperimentManifest
    run_dir: Path
    launch_target: LaunchTarget
    status: str
    steps: List[OrchestrationStep] = field(default_factory=list)
    reward_candidate: Optional[Dict[str, Any]] = None
    robustness_assessment: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": asdict(self.manifest),
            "run_dir": str(self.run_dir),
            "launch_target": self.launch_target.to_dict(),
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "reward_candidate": self.reward_candidate,
            "robustness_assessment": self.robustness_assessment,
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


class OrchestrationPolicy(Protocol):
    """Decision policy for the agentic loop."""

    def decide(self, state: OrchestrationState) -> OrchestrationDecision:
        """Return the next action label."""


class HeuristicOrchestrationPolicy:
    """Simple local policy used until the GPT-5.4-nano policy is wired in."""

    def decide(self, state: OrchestrationState) -> OrchestrationDecision:
        if state.reward_candidate_path is None:
            return OrchestrationDecision(
                decision="stop",
                rationale="No reward candidate was provided.",
                suggested_edit="Attach a reward candidate path before orchestrating.",
                suggested_patch=None,
                confidence=0.99,
            )
        if not state.validated:
            return OrchestrationDecision(
                decision="validate_reward_candidate",
                rationale="The candidate must be statically validated before launch planning.",
                suggested_edit="Check the candidate for disallowed imports or missing compute_reward.",
                suggested_patch="Ensure the module defines compute_reward(observation, action, info) and only imports allowed modules.",
                confidence=0.95,
            )
        if state.valid is False:
            return OrchestrationDecision(
                decision="stop",
                rationale="Validation failed, so the loop should stop.",
                suggested_edit="Fix the validation issues before previewing a launch.",
                suggested_patch="Patch the reward module to remove invalid imports or restore the required compute_reward signature.",
                confidence=0.98,
            )
        if not state.robustness_checked:
            return OrchestrationDecision(
                decision="assess_reward_robustness",
                rationale="Validation passed; a cheap robustness screen should run before launch planning.",
                suggested_edit="Review the robustness score and scenario plan before launch.",
                suggested_patch="If robustness is only medium or worse, tighten the shaping term or add bounded state-aware terms before launch.",
                confidence=0.92,
            )
        if state.robustness_rejected:
            return OrchestrationDecision(
                decision="stop",
                rationale="The robustness check rejected the candidate.",
                suggested_edit="Fix the robustness issues before previewing or scripting a launch.",
                suggested_patch="Rework the reward to reduce hacking risk or better align it with the task state structure.",
                confidence=0.97,
            )
        if not state.previewed:
            return OrchestrationDecision(
                decision="preview_launch",
                rationale="Validation passed; the execution plan should be previewed next.",
                suggested_edit="Review the environment, algorithm, and training steps in the plan.",
                suggested_patch="If the reward seems too sparse, add a small shaping term tied to the environment-specific signal before re-running preview.",
                confidence=0.9,
            )
        if not state.scripted:
            return OrchestrationDecision(
                decision="write_launch_script",
                rationale="The launch plan is ready to be written as a shell script.",
                suggested_edit="Generate the launch script for the GPU VM handoff.",
                suggested_patch="No reward code change needed; preserve the candidate and write the launch script so the current reward can be executed unchanged.",
                confidence=0.88,
            )
        return OrchestrationDecision(
            decision="stop",
            rationale="All requested tools have completed for this pass.",
            suggested_edit=None,
            suggested_patch=None,
            confidence=0.9,
        )


class OpenAIOrchestrationPolicy:
    """GPT-5.4-nano-backed policy using the Responses API.

    Falls back to a heuristic policy when the OpenAI SDK or API key is unavailable.
    """

    DECISION_SCHEMA = {
        "type": "json_schema",
        "name": "orchestration_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [
                        "stop",
                        "validate_reward_candidate",
                        "assess_reward_robustness",
                        "preview_launch",
                        "write_launch_script",
                    ],
                },
                "rationale": {
                    "type": "string",
                },
                "suggested_edit": {
                    "type": ["string", "null"],
                },
                "suggested_patch": {
                    "type": ["string", "null"],
                },
                "confidence": {
                    "type": ["number", "null"],
                },
            },
            "required": ["decision", "rationale"],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        model: Optional[str] = None,
        fallback: Optional[OrchestrationPolicy] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        self.fallback = fallback or HeuristicOrchestrationPolicy()
        self.client = client or self._build_client()

    @classmethod
    def from_env(cls, fallback: Optional[OrchestrationPolicy] = None) -> OrchestrationPolicy:
        if not os.getenv("OPENAI_API_KEY"):
            return fallback or HeuristicOrchestrationPolicy()
        try:
            return cls(fallback=fallback)
        except Exception:
            return fallback or HeuristicOrchestrationPolicy()

    def decide(self, state: OrchestrationState) -> OrchestrationDecision:
        if self.client is None:
            return self.fallback.decide(state)
        try:
            decision = self._decide_with_model(state)
        except Exception:
            return self.fallback.decide(state)
        return decision

    def _build_client(self) -> Optional[Any]:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None
        return OpenAI()

    def _decide_with_model(self, state: OrchestrationState) -> OrchestrationDecision:
        payload = self._build_payload(state)
        response = self.client.responses.create(
            model=self.model,
            instructions=self._instructions(),
            input=json.dumps(payload, indent=2, sort_keys=True),
            text={"format": self.DECISION_SCHEMA},
        )
        data = json.loads(response.output_text)
        return OrchestrationDecision(
            decision=str(data["decision"]),
            rationale=str(data["rationale"]),
            suggested_edit=data.get("suggested_edit"),
            suggested_patch=data.get("suggested_patch"),
            confidence=data.get("confidence"),
        )

    def _instructions(self) -> str:
        return (
            "You are the orchestration policy for a reward-engineering research prototype. "
            "Choose the next single action based on the current state. "
            "Prefer the smallest safe next step. "
            "Return JSON only."
        )

    def _build_payload(self, state: OrchestrationState) -> Dict[str, Any]:
        return {
            "model": self.model,
            "manifest": asdict(state.manifest),
            "run_dir": str(state.run_dir),
            "launch_target": state.launch_target.to_dict(),
            "reward_candidate_path": str(state.reward_candidate_path) if state.reward_candidate_path else None,
            "reward_entrypoint": state.reward_entrypoint,
            "validated": state.validated,
            "valid": state.valid,
            "robustness_checked": state.robustness_checked,
            "robustness_rejected": state.robustness_rejected,
            "robustness_score": state.robustness_score,
            "robustness_risk": state.robustness_risk,
            "previewed": state.previewed,
            "scripted": state.scripted,
            "current_step": state.current_step,
            "max_steps": state.max_steps,
        }


class AgenticOrchestrator:
    """Tool-driven loop for reward validation and launch planning."""

    def __init__(
        self,
        runner: Optional[ExperimentRunner] = None,
        policy: Optional[OrchestrationPolicy] = None,
        validator: Optional[RewardCandidateValidator] = None,
        loader: Optional[RewardCandidateLoader] = None,
    ) -> None:
        self.runner = runner
        self.policy = policy or OpenAIOrchestrationPolicy.from_env()
        self.validator = validator or RewardCandidateValidator()
        self.loader = loader or RewardCandidateLoader(self.validator)

    def run(
        self,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
        reward_candidate_path: Optional[Path] = None,
        reward_entrypoint: str = "compute_reward",
    ) -> OrchestrationTrace:
        runner = self.runner or ExperimentRunner(ProjectPaths(launch_target.working_directory))
        state = OrchestrationState(
            manifest=manifest,
            run_dir=run_dir,
            launch_target=launch_target,
            reward_candidate_path=reward_candidate_path,
            reward_entrypoint=reward_entrypoint,
        )
        steps: List[OrchestrationStep] = []
        reward_candidate_record: Optional[Dict[str, Any]] = manifest.reward_candidate
        robustness_assessment_record: Optional[Dict[str, Any]] = None
        notes: Optional[str] = None

        while state.current_step < state.max_steps:
            decision = self.policy.decide(state)
            if decision.decision == "stop":
                notes = notes or decision.rationale
                break

            if decision.decision == "validate_reward_candidate":
                result = self._validate_reward_candidate(state)
                reward_candidate_record = result.payload.get("reward_candidate")
                state.validated = True
                state.valid = result.payload.get("valid")
            elif decision.decision == "assess_reward_robustness":
                result = self._assess_reward_robustness(state)
                robustness_assessment_record = result.payload.get("robustness_assessment")
                state.robustness_checked = True
                state.robustness_rejected = bool(result.payload.get("should_reject"))
                state.robustness_score = result.payload.get("overall_score")
                state.robustness_risk = result.payload.get("risk_level")
                if state.robustness_rejected:
                    notes = result.message or notes
            elif decision.decision == "preview_launch":
                result = self._preview_launch(runner, manifest, run_dir, launch_target)
                state.previewed = True
            elif decision.decision == "write_launch_script":
                result = self._write_launch_script(runner, manifest, run_dir, launch_target)
                state.scripted = True
            else:
                result = OrchestrationToolResult(
                    tool_name="unknown",
                    status="error",
                    payload={},
                    message="Unsupported decision: {0}".format(decision.decision),
                )
                notes = result.message
                steps.append(OrchestrationStep(index=state.current_step, decision=decision, result=result))
                break

            steps.append(OrchestrationStep(index=state.current_step, decision=decision, result=result))
            if notes is None and decision.suggested_edit:
                notes = decision.suggested_edit
            state.current_step += 1

        final_status = (
            "completed"
            if state.reward_candidate_path
            and state.valid is not False
            and state.robustness_checked
            and not state.robustness_rejected
            and state.scripted
            else "stopped"
        )
        if state.reward_candidate_path is None:
            final_status = "idle"
            notes = notes or "No reward candidate provided."
        elif state.robustness_rejected:
            final_status = "stopped"
            notes = notes or "Reward candidate rejected by robustness assessment."

        return OrchestrationTrace(
            manifest=manifest,
            run_dir=run_dir,
            launch_target=launch_target,
            status=final_status,
            steps=steps,
            reward_candidate=reward_candidate_record,
            robustness_assessment=robustness_assessment_record,
            notes=notes,
        )

    def _validate_reward_candidate(self, state: OrchestrationState) -> OrchestrationToolResult:
        if state.reward_candidate_path is None:
            return OrchestrationToolResult(
                tool_name="validate_reward_candidate",
                status="skipped",
                message="No reward candidate path provided.",
            )

        loaded = self.loader.load(state.reward_candidate_path, entrypoint=state.reward_entrypoint)
        validation = self.validator.validate_file(state.reward_candidate_path, entrypoint=state.reward_entrypoint)
        reward_candidate_record = {
            "name": loaded.spec.name,
            "path": str(loaded.spec.path),
            "entrypoint": loaded.spec.entrypoint,
            "validation": validation.to_dict(),
        }
        state.manifest.reward_candidate = reward_candidate_record
        return OrchestrationToolResult(
            tool_name="validate_reward_candidate",
            status="validated",
            payload={
                "valid": validation.valid,
                "reward_candidate": reward_candidate_record,
                "issues": validation.to_dict()["issues"],
            },
            message="Reward candidate validated successfully." if validation.valid else "Reward candidate invalid.",
        )

    def _assess_reward_robustness(self, state: OrchestrationState) -> OrchestrationToolResult:
        if state.reward_candidate_path is None:
            return OrchestrationToolResult(
                tool_name="assess_reward_robustness",
                status="skipped",
                message="No reward candidate path provided.",
            )

        assessment = RewardRobustnessAnalyzer().assess_file(
            state.reward_candidate_path,
            entrypoint=state.reward_entrypoint,
        )
        return OrchestrationToolResult(
            tool_name="assess_reward_robustness",
            status="assessed",
            payload={
                "overall_score": assessment.overall_score,
                "risk_level": assessment.risk_level,
                "should_reject": assessment.should_reject,
                "robustness_assessment": assessment.to_dict(),
            },
            message=assessment.summary,
        )

    def _preview_launch(
        self,
        runner: ExperimentRunner,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> OrchestrationToolResult:
        receipt = runner.preview_ppo_launch(manifest, run_dir, launch_target)
        return OrchestrationToolResult(
            tool_name="preview_launch",
            status=receipt.status,
            payload=receipt.to_dict(),
            message=receipt.notes,
        )

    def _write_launch_script(
        self,
        runner: ExperimentRunner,
        manifest: ExperimentManifest,
        run_dir: Path,
        launch_target: LaunchTarget,
    ) -> OrchestrationToolResult:
        receipt = runner.write_ppo_launch_script(manifest, run_dir, launch_target)
        return OrchestrationToolResult(
            tool_name="write_launch_script",
            status=receipt.status,
            payload=receipt.to_dict(),
            message=receipt.notes,
        )
