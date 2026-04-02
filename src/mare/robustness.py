from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast
import json
import re


@dataclass(frozen=True)
class RobustnessScenario:
    """A low-cost robustness probe for a reward candidate."""

    name: str
    description: str
    expected_risk: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessSignal:
    """A single heuristic signal used to score reward robustness."""

    name: str
    score: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessAssessment:
    """Heuristic assessment for reward-hacking and overfitting risk."""

    path: Path
    environment: Optional[str]
    valid: bool
    overall_score: float
    risk_level: str
    should_reject: bool
    summary: str
    signals: List[RobustnessSignal] = field(default_factory=list)
    scenarios: List[RobustnessScenario] = field(default_factory=list)
    trace_path: Optional[Path] = None
    latest_step: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "environment": self.environment,
            "valid": self.valid,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "should_reject": self.should_reject,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "latest_step": self.latest_step,
        }


@dataclass(frozen=True)
class RunSummary:
    """Compact run-level summary combining orchestration and robustness state."""

    trace_path: Path
    orchestration_status: str
    step_count: int
    final_decision: Optional[str]
    reward_candidate_path: Optional[str]
    robustness_assessment: Optional[Dict[str, Any]]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_path": str(self.trace_path),
            "orchestration_status": self.orchestration_status,
            "step_count": self.step_count,
            "final_decision": self.final_decision,
            "reward_candidate_path": self.reward_candidate_path,
            "robustness_assessment": self.robustness_assessment,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Phase5ComparisonReport:
    """Combined view of orchestration, robustness, and sweep planning."""

    trace_summary: RunSummary
    sweep_plan_path: Optional[Path]
    sweep_plan: Optional[Dict[str, Any]]
    orchestration_vs_sweep_aligned: Optional[bool]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_summary": self.trace_summary.to_dict(),
            "sweep_plan_path": str(self.sweep_plan_path) if self.sweep_plan_path is not None else None,
            "sweep_plan": self.sweep_plan,
            "orchestration_vs_sweep_aligned": self.orchestration_vs_sweep_aligned,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ReviewBrief:
    """Human-readable Phase 5 review brief."""

    title: str
    lines: List[str]

    def to_text(self) -> str:
        return "\n".join([self.title, *[f"- {line}" for line in self.lines]]) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "lines": self.lines,
        }


@dataclass(frozen=True)
class ReadinessStatus:
    """One-line readiness status for a Phase 5 trace."""

    ready: bool
    label: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardRobustnessAnalyzer:
    """Cheap static robustness checks for reward candidates.

    The analyzer is intentionally lightweight so the project can run on a limited
    compute budget before IsaacGym execution is wired in.
    """

    def assess_file(self, path: Path, entrypoint: str = "compute_reward") -> RobustnessAssessment:
        source = path.read_text(encoding="utf-8")
        environment = self._infer_environment_from_source(source)
        valid = self._has_required_entrypoint(source, entrypoint)
        signals = self._score_signals(source, environment)
        overall_score = self._aggregate_score(signals, valid)
        risk_level = self._risk_level(overall_score, valid)
        should_reject = risk_level in {"high", "critical"} or not valid
        summary = self._build_summary(path, environment, overall_score, risk_level, valid)
        return RobustnessAssessment(
            path=path,
            environment=environment,
            valid=valid,
            overall_score=overall_score,
            risk_level=risk_level,
            should_reject=should_reject,
            summary=summary,
            signals=signals,
            scenarios=self._build_scenarios(environment),
        )

    def assess_trace(self, trace_path: Path) -> RobustnessAssessment:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest = trace_data.get("manifest", {})
        reward_candidate = manifest.get("reward_candidate") or trace_data.get("reward_candidate") or {}
        candidate_path = self._resolve_candidate_path(Path(reward_candidate.get("path", "")), trace_path)
        assessment = self.assess_file(candidate_path, entrypoint=str(reward_candidate.get("entrypoint", "compute_reward")))
        latest_step = self._latest_step(trace_data)
        environment = self._latest_environment(latest_step) or assessment.environment
        trace_signals = list(assessment.signals)
        trace_signals.extend(self._trace_signals(latest_step))
        overall_score = self._aggregate_score(trace_signals, assessment.valid)
        risk_level = self._risk_level(overall_score, assessment.valid)
        should_reject = risk_level in {"high", "critical"} or not assessment.valid
        summary = self._build_trace_summary(candidate_path, environment, overall_score, risk_level, latest_step)
        return RobustnessAssessment(
            path=candidate_path,
            environment=environment,
            valid=assessment.valid,
            overall_score=overall_score,
            risk_level=risk_level,
            should_reject=should_reject,
            summary=summary,
            signals=trace_signals,
            scenarios=self._build_scenarios(environment),
            trace_path=trace_path,
            latest_step=latest_step,
        )

    def summarize_trace(self, trace_path: Path) -> RunSummary:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        steps = trace_data.get("steps", [])
        final_decision = None
        if isinstance(steps, list) and steps:
            latest_step = steps[-1]
            if isinstance(latest_step, dict):
                final_decision = str(latest_step.get("decision", {}).get("decision")) or None
        manifest = trace_data.get("manifest", {})
        reward_candidate = manifest.get("reward_candidate") or trace_data.get("reward_candidate") or {}
        assessment = trace_data.get("robustness_assessment")
        if assessment is None:
            assessment = self.assess_trace(trace_path).to_dict()
        return RunSummary(
            trace_path=trace_path,
            orchestration_status=str(trace_data.get("status", "unknown")),
            step_count=len(steps) if isinstance(steps, list) else 0,
            final_decision=final_decision,
            reward_candidate_path=reward_candidate.get("path"),
            robustness_assessment=assessment,
            notes=trace_data.get("notes"),
        )

    def compare_trace_with_sweep(self, trace_path: Path, sweep_plan_path: Optional[Path] = None) -> Phase5ComparisonReport:
        trace_summary = self.summarize_trace(trace_path)
        if sweep_plan_path is None:
            candidate = trace_path.parent / "sweep_plan.json"
            sweep_plan_path = candidate if candidate.exists() else None

        sweep_plan = None
        if sweep_plan_path is not None and sweep_plan_path.exists():
            sweep_plan = json.loads(sweep_plan_path.read_text(encoding="utf-8"))

        orchestration_vs_sweep_aligned = None
        if sweep_plan is not None:
            orchestration_vs_sweep_aligned = self._compare_sweep_alignment(trace_summary, sweep_plan)

        summary = self._build_phase5_summary(trace_summary, sweep_plan, orchestration_vs_sweep_aligned)
        return Phase5ComparisonReport(
            trace_summary=trace_summary,
            sweep_plan_path=sweep_plan_path,
            sweep_plan=sweep_plan,
            orchestration_vs_sweep_aligned=orchestration_vs_sweep_aligned,
            summary=summary,
        )

    def build_review_brief(self, trace_path: Path, sweep_plan_path: Optional[Path] = None) -> ReviewBrief:
        report = self.compare_trace_with_sweep(trace_path, sweep_plan_path=sweep_plan_path)
        trace = report.trace_summary
        robustness = trace.robustness_assessment or {}
        lines = [
            f"Trace status: {trace.orchestration_status}",
            f"Steps completed: {trace.step_count}",
            f"Final decision: {trace.final_decision or 'unknown'}",
            f"Reward candidate: {trace.reward_candidate_path or 'none'}",
        ]
        if robustness:
            lines.append(
                "Robustness: {0} (score {1})".format(
                    robustness.get("risk_level", "unknown"),
                    robustness.get("overall_score", "n/a"),
                )
            )
            lines.append(
                "Robustness action: {0}".format(
                    "reject" if robustness.get("should_reject") else "keep reviewing"
                )
            )
        if report.sweep_plan is not None:
            lines.append(
                "Sweep plan: {0} runs, alignment={1}".format(
                    len(report.sweep_plan.get("runs", [])),
                    "matched" if report.orchestration_vs_sweep_aligned else "mismatch" if report.orchestration_vs_sweep_aligned is False else "unknown",
                )
            )
        else:
            lines.append("Sweep plan: not found")
        if trace.notes:
            lines.append(f"Notes: {trace.notes}")
        lines.append(
            "Recommendation: {0}".format(
                "candidate is not ready for expensive VM runs"
                if robustness.get("should_reject")
                else "candidate is reasonable for the next local planning step"
            )
        )
        return ReviewBrief(
            title="Phase 5 Review Brief",
            lines=lines,
        )

    def build_readiness_status(self, trace_path: Path, sweep_plan_path: Optional[Path] = None) -> ReadinessStatus:
        report = self.compare_trace_with_sweep(trace_path, sweep_plan_path=sweep_plan_path)
        trace = report.trace_summary
        robustness = trace.robustness_assessment or {}
        ready = bool(
            trace.orchestration_status == "completed"
            and trace.final_decision == "write_launch_script"
            and not robustness.get("should_reject")
            and report.orchestration_vs_sweep_aligned is not False
        )
        label = "READY" if ready else "HOLD"
        detail = (
            "trace={0}; robustness={1}; sweep={2}".format(
                trace.orchestration_status,
                robustness.get("risk_level", "unknown"),
                "aligned" if report.orchestration_vs_sweep_aligned is not False else "mismatch",
            )
        )
        return ReadinessStatus(
            ready=ready,
            label=label,
            detail=detail,
        )

    def _score_signals(self, source: str, environment: Optional[str]) -> List[RobustnessSignal]:
        signals: List[RobustnessSignal] = []
        return_expr = self._extract_return_expression(source)
        source_lower = source.lower()
        has_action = "action" in return_expr
        has_info = "info" in source_lower
        has_clamp = any(token in return_expr for token in ("min(", "max(", "clip", "clamp"))
        has_position = "position" in source_lower
        has_velocity = "velocity" in source_lower
        has_reward_constant = bool(re.search(r"return\s+([0-9]+(?:\.[0-9]+)?)", source))

        signals.append(
            RobustnessSignal(
                name="state_alignment",
                score=0.92 if has_position or has_velocity else 0.65,
                message="Reward references state features in the returned expression." if has_position or has_velocity else "Reward uses limited state information; verify it is not too sparse.",
            )
        )
        signals.append(
            RobustnessSignal(
                name="action_usage",
                score=0.88 if has_action else 0.55,
                message="Action appears in the reward expression." if has_action else "Action does not appear in the reward expression.",
            )
        )
        signals.append(
            RobustnessSignal(
                name="info_usage",
                score=0.8 if has_info else 0.6,
                message="Info is referenced in the source." if has_info else "Info is not referenced; that is acceptable but reduces context sensitivity.",
            )
        )
        signals.append(
            RobustnessSignal(
                name="bounded_shaping",
                score=0.94 if has_clamp else 0.58,
                message="Reward shaping uses bounded terms." if has_clamp else "Reward shaping is unbounded; consider clipping or min/max guards.",
            )
        )
        signals.append(
            RobustnessSignal(
                name="sparse_reward_risk",
                score=0.9 if not has_reward_constant else 0.72,
                message="Reward is shaped beyond a raw constant return." if not has_reward_constant else "Reward may be too close to a constant baseline.",
            )
        )
        signals.append(
            RobustnessSignal(
                name="environment_fit",
                score=self._environment_fit_score(environment, source_lower),
                message=self._environment_fit_message(environment),
            )
        )
        return signals

    def _trace_signals(self, latest_step: Optional[Dict[str, Any]]) -> List[RobustnessSignal]:
        if latest_step is None:
            return [
                RobustnessSignal(
                    name="trace_context",
                    score=0.5,
                    message="No orchestration step was available in the trace.",
                )
            ]
        decision = str(latest_step.get("decision", {}).get("decision", ""))
        tool_name = str(latest_step.get("result", {}).get("tool_name", ""))
        if decision == "write_launch_script" and tool_name == "write_launch_script":
            return [
                RobustnessSignal(
                    name="trace_stage",
                    score=0.82,
                    message="Trace reached launch script generation, so reward validation and preview were both completed.",
                )
            ]
        if decision == "preview_launch":
            return [
                RobustnessSignal(
                    name="trace_stage",
                    score=0.78,
                    message="Trace reached launch preview but not script generation.",
                )
            ]
        return [
            RobustnessSignal(
                name="trace_stage",
                score=0.7,
                message="Trace shows an incomplete orchestration pass.",
            )
        ]

    def _build_scenarios(self, environment: Optional[str]) -> List[RobustnessScenario]:
        return [
            RobustnessScenario(
                name="seed_shift",
                description="Re-run with a nearby seed to detect instability across random initializations.",
                expected_risk="Low if reward is stable; medium if the candidate overfits to initialization noise.",
                config={"seeds": [-1, 1]},
            ),
            RobustnessScenario(
                name="hyperparameter_perturbation",
                description="Try a very small learning-rate and clip-range perturbation to expose reward sensitivity.",
                expected_risk="Low if the reward is well-shaped; high if performance collapses under minor changes.",
                config={"learning_rate_scale": [0.5, 1.5], "clip_range_scale": [0.8, 1.2]},
            ),
            RobustnessScenario(
                name="environment_alignment",
                description="Check whether the candidate's shaping terms match the environment's state structure.",
                expected_risk="Low when the reward references the right state variables for the task.",
                config={"environment": environment},
            ),
        ]

    def _aggregate_score(self, signals: List[RobustnessSignal], valid: bool) -> float:
        if not signals:
            return 0.0
        average = sum(signal.score for signal in signals) / float(len(signals))
        if not valid:
            average *= 0.5
        return round(max(0.0, min(1.0, average)), 3)

    def _risk_level(self, score: float, valid: bool) -> str:
        if not valid:
            return "critical"
        if score >= 0.82:
            return "low"
        if score >= 0.68:
            return "medium"
        if score >= 0.5:
            return "high"
        return "critical"

    def _build_summary(self, path: Path, environment: Optional[str], score: float, risk_level: str, valid: bool) -> str:
        env_text = environment or "unknown environment"
        status_text = "valid" if valid else "invalid"
        return (
            f"{path.name} is {status_text} for {env_text}; robustness score={score:.3f}, risk={risk_level}."
        )

    def _build_trace_summary(
        self,
        path: Path,
        environment: Optional[str],
        score: float,
        risk_level: str,
        latest_step: Optional[Dict[str, Any]],
    ) -> str:
        decision = str(latest_step.get("decision", {}).get("decision", "")) if latest_step else "none"
        env_text = environment or "unknown environment"
        return (
            f"{path.name} assessed from trace for {env_text}; latest decision={decision}, robustness score={score:.3f}, "
            f"risk={risk_level}."
        )

    def _infer_environment_from_source(self, source: str) -> Optional[str]:
        lowered = source.lower()
        if "cartpole" in lowered:
            return "CartPole"
        if "humanoid" in lowered:
            return "Humanoid"
        if "allegro" in lowered:
            return "AllegroHand"
        return None

    def _environment_fit_score(self, environment: Optional[str], source_lower: str) -> float:
        if environment == "CartPole":
            return 0.96 if "position" in source_lower and "velocity" in source_lower else 0.78
        if environment == "Humanoid":
            return 0.9 if "observation" in source_lower else 0.74
        if environment == "AllegroHand":
            return 0.9 if "observation" in source_lower else 0.72
        return 0.7

    def _environment_fit_message(self, environment: Optional[str]) -> str:
        if environment is None:
            return "Environment could not be inferred from the candidate source."
        return f"Candidate appears aligned with {environment}."

    def _extract_return_expression(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
                for child in node.body:
                    if isinstance(child, ast.Return) and child.value is not None:
                        return ast.unparse(child.value) if hasattr(ast, "unparse") else ""
        return ""

    def _has_required_entrypoint(self, source: str, entrypoint: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
                args = node.args
                return len(args.args) == 3 and not args.vararg and not args.kwonlyargs and not args.kwarg
        return False

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

    def _compare_sweep_alignment(self, trace_summary: RunSummary, sweep_plan: Dict[str, Any]) -> Optional[bool]:
        trace_environment = trace_summary.robustness_assessment.get("environment") if trace_summary.robustness_assessment else None
        sweep_environment = sweep_plan.get("environment")
        if trace_environment is None or sweep_environment is None:
            return None
        return str(trace_environment) == str(sweep_environment)

    def _build_phase5_summary(
        self,
        trace_summary: RunSummary,
        sweep_plan: Optional[Dict[str, Any]],
        orchestration_vs_sweep_aligned: Optional[bool],
    ) -> str:
        parts = [
            f"orchestration={trace_summary.orchestration_status}",
            f"steps={trace_summary.step_count}",
        ]
        if trace_summary.robustness_assessment is not None:
            robustness = trace_summary.robustness_assessment
            parts.append(
                "robustness={0}:{1}".format(
                    robustness.get("risk_level", "unknown"),
                    robustness.get("overall_score", "n/a"),
                )
            )
        if sweep_plan is not None:
            parts.append(f"sweep_runs={len(sweep_plan.get('runs', []))}")
            if orchestration_vs_sweep_aligned is True:
                parts.append("alignment=matched")
            elif orchestration_vs_sweep_aligned is False:
                parts.append("alignment=mismatched")
            else:
                parts.append("alignment=unknown")
        else:
            parts.append("sweep_runs=none")
        return "; ".join(parts)
