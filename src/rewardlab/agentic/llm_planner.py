"""
Summary: LLM-driven primary decision planner for agentic tool-calling runs.
Created: 2026-04-11
Last Updated: 2026-04-11
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.stop_guidance import StopGuidanceDecision
from rewardlab.llm.openai_client import OpenAIClient, OpenAIClientConfig, OpenAITextResponse
from rewardlab.schemas.agentic_run import AgentDecision, AgentDecisionAction, AgenticRunSpec
from rewardlab.schemas.budget_state import BudgetState


class PlannerClient(Protocol):
    """
    Define the text-generation interface required by the primary planner.
    """

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Generate one planner response from prompt text.
        """


@dataclass(slots=True, frozen=True)
class PlannerCallUsage:
    """
    Capture normalized planner-call usage for budget accounting.
    """

    model_used: str | None
    api_input_tokens: int = 0
    api_output_tokens: int = 0
    api_cost_usd: float = 0.0
    call_count: int = 1


@dataclass(slots=True, frozen=True)
class PlannerAttemptFeedback:
    """
    Capture one failed planner attempt for trace and debugging visibility.
    """

    attempt_index: int
    max_attempts: int
    failure_type: str
    reason: str
    output_excerpt: str | None = None


class DecisionPlanner(Protocol):
    """
    Define the planner interface consumed by the primary optimizer.
    """

    def plan(
        self,
        *,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
        stop_hint: StopGuidanceDecision | None,
    ) -> AgentDecision | None:
        """
        Return one decision, or `None` when planner is unavailable.
        """

    def last_usage(self) -> PlannerCallUsage | None:
        """
        Return usage from the latest planner attempt when available.
        """

    def last_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return failure feedback rows from the latest planner call.
        """


class LLMDecisionPlanner:
    """
    Build context-rich prompts and parse one-turn decisions from an LLM.
    """

    _TOOL_DESCRIPTIONS: dict[str, str] = {
        "run_experiment": "Run a candidate reward/environment experiment and return score metrics.",
        "run_probe_suite": "Run robustness probes for one candidate and return risk analysis.",
        (
            "compare_candidates"
        ): "Rank candidate snapshots and choose current best aggregate candidate.",
        "export_report": "Persist a report payload to a JSON artifact path.",
        "budget_snapshot": "Return remaining budget dimensions.",
        "read_artifact": "Read a bounded preview from a local artifact path.",
    }
    _TOOL_ARGUMENT_CONTRACTS: dict[str, dict[str, str]] = {
        "run_experiment": {
            "required": (
                "environment_id, environment_backend, objective_file, reward_file, "
                "session_id, candidate_id, seed, iteration_index, variant_label, "
                "include_reflection, overrides"
            ),
            "optional": "none",
        },
        "run_probe_suite": {
            "required": (
                "candidate_id, primary_score, environment_id, environment_backend, "
                "objective_file, reward_file, session_id, seed, iteration_index, "
                "variant_label, overrides"
            ),
            "optional": "none",
        },
        "compare_candidates": {
            "required": "candidates",
            "optional": "none",
        },
        "export_report": {
            "required": "report_payload",
            "optional": "output_path",
        },
        "budget_snapshot": {
            "required": "none",
            "optional": "none",
        },
        "read_artifact": {
            "required": "path",
            "optional": "max_chars",
        },
    }

    def __init__(
        self,
        *,
        client_factory: Callable[[str], PlannerClient] | None = None,
    ) -> None:
        """
        Initialize planner with injectable client-factory dependency.
        """
        self._client_factory = client_factory or self._default_client_factory
        self._last_usage: PlannerCallUsage | None = None
        self._last_feedback: tuple[PlannerAttemptFeedback, ...] = ()

    def plan(
        self,
        *,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
        stop_hint: StopGuidanceDecision | None,
    ) -> AgentDecision | None:
        """
        Return an LLM-authored decision, or `None` when planner is unavailable.
        """
        self._last_usage = None
        feedback_rows: list[PlannerAttemptFeedback] = []
        provider = spec.agent.planner_provider.strip().lower()
        if provider != "openai":
            self._last_feedback = ()
            return None

        base_prompt = self._build_prompt(
            turn_index=turn_index,
            spec=spec,
            context=context,
            budget_state=budget_state,
            stop_hint=stop_hint,
        )
        try:
            client = self._client_factory(spec.agent.primary_model)
        except Exception:  # noqa: BLE001
            self._last_feedback = ()
            return None

        max_attempts = 1 + max(0, spec.agent.planner_max_retries)
        allowed_tools = set(spec.tools.enabled)
        total_usage: PlannerCallUsage | None = None
        previous_output: str | None = None
        repair_reason = "initial planner attempt"
        for attempt_index in range(max_attempts):
            prompt = (
                base_prompt
                if attempt_index == 0
                else self._build_repair_prompt(
                    base_prompt=base_prompt,
                    reason=repair_reason,
                    previous_output=previous_output,
                )
            )
            try:
                response_text, usage = self._generate_text_with_usage(
                    client=client,
                    prompt=prompt,
                    max_output_tokens=spec.agent.planner_max_output_tokens,
                    reasoning_effort=spec.agent.reasoning_effort.value,
                    default_model=spec.agent.primary_model,
                )
                total_usage = _accumulate_usage(total_usage=total_usage, usage=usage)
            except Exception:  # noqa: BLE001
                repair_reason = "planner request failed"
                previous_output = None
                feedback_rows.append(
                    PlannerAttemptFeedback(
                        attempt_index=attempt_index + 1,
                        max_attempts=max_attempts,
                        failure_type="request_failed",
                        reason=repair_reason,
                        output_excerpt=None,
                    )
                )
                continue

            previous_output = response_text
            payload = self._extract_json_payload(response_text)
            if payload is None:
                repair_reason = "response was not a valid JSON object"
                feedback_rows.append(
                    PlannerAttemptFeedback(
                        attempt_index=attempt_index + 1,
                        max_attempts=max_attempts,
                        failure_type="parse_error",
                        reason=repair_reason,
                        output_excerpt=_output_excerpt(response_text),
                    )
                )
                continue
            payload["turn_index"] = turn_index
            payload["decision_source"] = "llm_openai"
            try:
                decision = AgentDecision.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                repair_reason = f"decision schema validation failed: {exc.__class__.__name__}"
                feedback_rows.append(
                    PlannerAttemptFeedback(
                        attempt_index=attempt_index + 1,
                        max_attempts=max_attempts,
                        failure_type="schema_validation_error",
                        reason=repair_reason,
                        output_excerpt=_output_excerpt(response_text),
                    )
                )
                continue
            if (
                decision.action is AgentDecisionAction.REQUEST_TOOL
                and decision.tool_name not in allowed_tools
            ):
                repair_reason = (
                    "requested tool is not enabled: "
                    f"{decision.tool_name}"
                )
                feedback_rows.append(
                    PlannerAttemptFeedback(
                        attempt_index=attempt_index + 1,
                        max_attempts=max_attempts,
                        failure_type="tool_not_enabled",
                        reason=repair_reason,
                        output_excerpt=_output_excerpt(response_text),
                    )
                )
                continue
            self._last_usage = total_usage
            self._last_feedback = tuple(feedback_rows)
            return decision

        self._last_usage = total_usage
        self._last_feedback = tuple(feedback_rows)
        return None

    def last_usage(self) -> PlannerCallUsage | None:
        """
        Return usage from the latest planner attempt when available.
        """
        return self._last_usage

    def last_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return failure feedback rows from the latest planner call.
        """
        return self._last_feedback

    @staticmethod
    def _default_client_factory(model: str) -> PlannerClient:
        """
        Build the default OpenAI planner client for a specific model.
        """
        return OpenAIClient(OpenAIClientConfig(model=model))

    @staticmethod
    def _generate_text_with_usage(
        *,
        client: PlannerClient,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str,
        default_model: str,
    ) -> tuple[str, PlannerCallUsage]:
        """
        Request planner text and collect usage when client supports it.
        """
        usage_method = getattr(client, "generate_text_with_usage", None)
        if callable(usage_method):
            response = usage_method(
                prompt,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
            if isinstance(response, OpenAITextResponse):
                return response.text, PlannerCallUsage(
                    model_used=response.model_used or default_model,
                    api_input_tokens=response.api_input_tokens,
                    api_output_tokens=response.api_output_tokens,
                    api_cost_usd=response.api_cost_usd,
                    call_count=1,
                )
            text_from_usage = getattr(response, "text", None)
            if isinstance(text_from_usage, str) and text_from_usage.strip():
                model_used = getattr(response, "model_used", default_model)
                return text_from_usage, PlannerCallUsage(
                    model_used=model_used if isinstance(model_used, str) else default_model,
                    api_input_tokens=_as_non_negative_int(
                        getattr(response, "api_input_tokens", 0)
                    ),
                    api_output_tokens=_as_non_negative_int(
                        getattr(response, "api_output_tokens", 0)
                    ),
                    api_cost_usd=_as_non_negative_float(getattr(response, "api_cost_usd", 0.0)),
                    call_count=1,
                )

        text = client.generate_text(
            prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        return text, PlannerCallUsage(model_used=default_model, call_count=1)

    def _build_prompt(
        self,
        *,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
        stop_hint: StopGuidanceDecision | None,
    ) -> str:
        """
        Build one context-rich planner prompt for the primary optimization agent.
        """
        objective_excerpt = _read_text_excerpt(spec.objective.text_file, max_chars=2400)
        baseline_excerpt = _read_text_excerpt(spec.objective.baseline_reward_file, max_chars=1600)
        system_prompt_extra = ""
        if spec.agent.planner_system_prompt_file:
            system_prompt_extra = _read_text_excerpt(
                spec.agent.planner_system_prompt_file,
                max_chars=3000,
            )
        tool_lines = []
        for tool_name in spec.tools.enabled:
            description = self._TOOL_DESCRIPTIONS.get(tool_name, "No description provided.")
            tool_lines.append(
                f"- {tool_name}: {description}"
            )
        recent_context = self._compact_recent_context(
            context=context,
            window=spec.agent.planner_context_window,
        )
        stop_hint_text = "none"
        if stop_hint is not None:
            stop_hint_text = json.dumps(
                {
                    "tag": stop_hint.tag.value,
                    "reason": stop_hint.reason,
                    "summary": stop_hint.summary,
                },
                indent=2,
                sort_keys=True,
            )
        output_contract = {
            "action": "request_tool | stop",
            "summary": "one concise sentence",
            "tool_name": "required when action=request_tool",
            "tool_arguments": "object; required when action=request_tool",
            "tool_rationale": "required when action=request_tool",
            "stop_tag": "required when action=stop",
            "stop_reason": "required when action=stop",
        }
        return "\n\n".join(
            [
                (
                    "You are the primary optimization agent for a"
                    " reinforcement-learning reward-search run."
                    " Decide the single best next action."
                ),
                (
                    "Hard requirements:\n"
                    "- Return exactly one JSON object and nothing else.\n"
                    "- If action=request_tool, choose only from the enabled tool list.\n"
                    "- If action=stop, include a stop_tag and stop_reason.\n"
                    "- Prefer high-value tool calls under remaining budget.\n"
                    "- Avoid rigid loops; choose actions based on evidence trend, risk, and budget."
                ),
                (
                    "Run context:\n"
                    f"- turn_index: {turn_index}\n"
                    f"- run_name: {spec.run_name}\n"
                    f"- environment: {spec.environment.backend.value}/{spec.environment.id}\n"
                    f"- seed: {spec.environment.seed}\n"
                    f"- best_score: {context.best_score}\n"
                    f"- enabled_tools:\n" + "\n".join(tool_lines)
                ),
                (
                    "Soft guidance:\n"
                    f"- target_env_return: {spec.budgets.soft.target_env_return}\n"
                    f"- plateau_window_turns: {spec.budgets.soft.plateau_window_turns}\n"
                    f"- min_delta_return: {spec.budgets.soft.min_delta_return}\n"
                    f"- min_gain_per_1k_usd: {spec.budgets.soft.min_gain_per_1k_usd}\n"
                    f"- risk_ceiling: {spec.budgets.soft.risk_ceiling}"
                ),
                "Stop-hint from heuristic guardrail:\n" + stop_hint_text,
                "Remaining budget snapshot:\n" + json.dumps(
                    budget_state.remaining(), indent=2, sort_keys=True
                ),
                "Objective excerpt:\n" + objective_excerpt,
                "Baseline reward excerpt:\n" + baseline_excerpt,
                "Recent decision/tool context:\n"
                + json.dumps(recent_context, indent=2, sort_keys=True),
                (
                    "Required JSON output contract:\n"
                    + json.dumps(output_contract, indent=2, sort_keys=True)
                ),
                "Tool argument contracts:\n" + self._render_tool_contracts(spec.tools.enabled),
                ("Additional planner instructions:\n" + system_prompt_extra)
                if system_prompt_extra
                else "No additional planner instructions provided.",
            ]
        )

    def _render_tool_contracts(self, enabled_tools: tuple[str, ...]) -> str:
        """
        Render per-tool required and optional argument keys for planner guidance.
        """
        lines: list[str] = []
        for tool_name in enabled_tools:
            contract = self._TOOL_ARGUMENT_CONTRACTS.get(tool_name)
            if contract is None:
                lines.append(f"- {tool_name}: required=unknown optional=unknown")
                continue
            required = contract.get("required", "unknown")
            optional = contract.get("optional", "unknown")
            lines.append(f"- {tool_name}: required={required}; optional={optional}")
        return "\n".join(lines)

    @staticmethod
    def _build_repair_prompt(
        *,
        base_prompt: str,
        reason: str,
        previous_output: str | None,
    ) -> str:
        """
        Build one repair prompt after an invalid planner output attempt.
        """
        previous_excerpt = "[none]"
        if previous_output is not None:
            trimmed = previous_output.strip()
            if trimmed:
                previous_excerpt = trimmed[:1200]
        return "\n\n".join(
            [
                base_prompt,
                "Your previous output was invalid and must be corrected.",
                f"Invalid reason: {reason}",
                "Previous output excerpt:",
                previous_excerpt,
                (
                    "Return exactly one valid JSON object that satisfies the required "
                    "contract and tool argument rules."
                ),
            ]
        )

    @staticmethod
    def _compact_recent_context(
        *,
        context: ContextStore,
        window: int,
    ) -> dict[str, object]:
        """
        Build compact recent context slices safe to include in planner prompts.
        """
        recent_decisions = context.decisions[-window:]
        recent_results = context.tool_results[-window:]
        decisions_payload: list[dict[str, object]] = []
        for row in recent_decisions:
            decisions_payload.append(
                {
                    "turn_index": row.turn_index,
                    "action": row.action.value,
                    "summary": row.summary,
                    "tool_name": row.tool_name,
                    "stop_tag": row.stop_tag.value if row.stop_tag else None,
                    "stop_reason": row.stop_reason,
                }
            )
        results_payload: list[dict[str, object]] = []
        for result_row in recent_results:
            results_payload.append(
                {
                    "turn_index": result_row.turn_index,
                    "tool_name": result_row.tool_name,
                    "status": result_row.status.value,
                    "candidate_id": result_row.output.get("candidate_id"),
                    "score": result_row.output.get("score"),
                    "risk_level": result_row.output.get("risk_level"),
                    "error": result_row.error,
                }
            )
        return {
            "recent_decisions": decisions_payload,
            "recent_tool_results": results_payload,
        }

    @staticmethod
    def _extract_json_payload(response_text: str) -> dict[str, Any] | None:
        """
        Parse one JSON decision object from raw model output text.
        """
        candidates: list[str] = [response_text]
        stripped = response_text.strip()
        if "```" in stripped:
            for chunk in stripped.split("```"):
                candidate = chunk.strip()
                if not candidate:
                    continue
                if candidate.lower().startswith("json"):
                    candidates.insert(0, candidate[4:].strip())
                    continue
                if candidate.startswith("{"):
                    candidates.insert(0, candidate)
        for candidate in candidates:
            parsed = _try_parse_json_object(candidate)
            if parsed is not None:
                return parsed
        return None


def _read_text_excerpt(path_text: str, *, max_chars: int) -> str:
    """
    Read bounded UTF-8 file text for planner prompt context.
    """
    path = Path(path_text)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return f"[unavailable: {path}]"
    excerpt = raw.strip()
    if not excerpt:
        return f"[empty: {path}]"
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[:max_chars] + "\n... [truncated]"


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    """
    Attempt to parse one JSON object from raw candidate text.
    """
    trimmed = candidate.strip()
    if not trimmed:
        return None
    parsed = _parse_candidate_object(trimmed)
    if parsed is not None:
        return parsed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start < 0 or end <= start:
        return None
    return _parse_candidate_object(trimmed[start : end + 1])


def _parse_candidate_object(text: str) -> dict[str, Any] | None:
    """
    Parse a JSON text candidate and return dict payload when valid.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    payload: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(key, str):
            payload[key] = value
    return payload


def _as_non_negative_int(raw: object) -> int:
    """
    Coerce numeric usage fields into non-negative integers.
    """
    if not isinstance(raw, int | float):
        return 0
    return max(0, int(raw))


def _as_non_negative_float(raw: object) -> float:
    """
    Coerce numeric usage fields into non-negative floats.
    """
    if not isinstance(raw, int | float):
        return 0.0
    return max(0.0, float(raw))


def _accumulate_usage(
    *,
    total_usage: PlannerCallUsage | None,
    usage: PlannerCallUsage,
) -> PlannerCallUsage:
    """
    Merge one planner-call usage row into cumulative usage totals.
    """
    if total_usage is None:
        return usage
    model_used = usage.model_used or total_usage.model_used
    return PlannerCallUsage(
        model_used=model_used,
        api_input_tokens=total_usage.api_input_tokens + usage.api_input_tokens,
        api_output_tokens=total_usage.api_output_tokens + usage.api_output_tokens,
        api_cost_usd=total_usage.api_cost_usd + usage.api_cost_usd,
        call_count=total_usage.call_count + usage.call_count,
    )


def _output_excerpt(text: str, *, max_chars: int = 220) -> str:
    """
    Return a compact one-line excerpt from planner output text.
    """
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."
