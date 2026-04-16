"""
Summary: Reward-candidate generation strategies for offline and OpenAI-backed iteration.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, cast

from rewardlab.experiments.reward_program import load_reward_program
from rewardlab.llm.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import EnvironmentBackend
from rewardlab.utils.env import load_runtime_environment

GENERIC_REWARD_PARAMETER_NAMES = (
    "state",
    "observation",
    "previous_observation",
    "next_observation",
    "env_reward",
    "environment_reward",
    "terminated",
    "truncated",
    "action",
    "step_index",
    "info",
)


class RewardDesignerMode(StrEnum):
    """Supported reward-generation strategies."""

    DETERMINISTIC = "deterministic"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class RewardDesignConfig:
    """Runtime configuration for reward-candidate generation."""

    mode: RewardDesignerMode = RewardDesignerMode.DETERMINISTIC
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
    max_tokens: int = 2_000

    @classmethod
    def from_environment(cls) -> RewardDesignConfig:
        """Load reward-designer configuration from runtime environment variables."""

        env = load_runtime_environment()
        raw_mode = env.get("REWARDLAB_REWARD_DESIGN_MODE", RewardDesignerMode.DETERMINISTIC)
        try:
            mode = RewardDesignerMode(raw_mode)
        except ValueError as exc:
            raise ValueError(
                f"unsupported REWARDLAB_REWARD_DESIGN_MODE value: {raw_mode!r}"
            ) from exc

        reasoning_effort = _reasoning_effort_from_environment(
            env.get("REWARDLAB_REWARD_DESIGN_REASONING_EFFORT")
        )
        max_tokens = _int_from_environment(
            env.get("REWARDLAB_REWARD_DESIGN_MAX_TOKENS"),
            default=2_000,
        )
        return cls(
            mode=mode,
            model=env.get("REWARDLAB_REWARD_DESIGN_MODEL", "gpt-5-mini"),
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True, slots=True)
class RewardDesignRequest:
    """Structured context supplied to a reward designer for the next iteration."""

    session_id: str
    objective_text: str
    environment_id: str
    environment_backend: EnvironmentBackend
    current_candidate: RewardCandidate
    next_iteration_index: int
    latest_reflection: ReflectionRecord | None = None
    latest_run: ExperimentRun | None = None
    prior_candidates: tuple[RewardCandidate, ...] = ()
    recent_decisions: tuple[dict[str, Any], ...] = ()
    recent_feedback: tuple[dict[str, Any], ...] = ()
    recent_robustness_assessments: tuple[dict[str, Any], ...] = ()
    allowed_parameter_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewardDesignResult:
    """Normalized reward-designer output."""

    reward_definition: str
    change_summary: str
    proposed_changes: list[str]
    model_name: str | None = None


class RewardDesigner(Protocol):
    """Protocol implemented by reward-generation strategies."""

    mode: RewardDesignerMode

    def design_next_candidate(
        self,
        request: RewardDesignRequest,
    ) -> RewardDesignResult:
        """Return the next reward candidate for a session."""


class DeterministicRewardDesigner:
    """Offline-safe reward designer that applies the legacy local rewrite heuristic."""

    mode = RewardDesignerMode.DETERMINISTIC

    def design_next_candidate(
        self,
        request: RewardDesignRequest,
    ) -> RewardDesignResult:
        """Generate a stable local revision without any network dependency."""

        proposed_changes = [
            "Increase reward emphasis on smooth, centered behavior.",
            "Add clearer stability incentives aligned with the objective.",
        ]
        revised_reward_definition = _revise_reward_definition(
            request.current_candidate.reward_definition,
            objective_text=request.objective_text,
            proposed_changes=proposed_changes,
            iteration_index=request.next_iteration_index,
        )
        return RewardDesignResult(
            reward_definition=revised_reward_definition,
            change_summary="Revision generated from deterministic reflection feedback.",
            proposed_changes=proposed_changes,
        )


class OpenAIRewardDesigner:
    """Reward designer that delegates candidate generation to an OpenAI chat model."""

    mode = RewardDesignerMode.OPENAI

    def __init__(
        self,
        *,
        openai_client: OpenAIClient | None = None,
        config: RewardDesignConfig | None = None,
    ) -> None:
        """Store the injected client plus explicit runtime configuration."""

        resolved_config = config or RewardDesignConfig.from_environment()
        self.config = resolved_config
        self.openai_client = openai_client or OpenAIClient()

    def design_next_candidate(
        self,
        request: RewardDesignRequest,
    ) -> RewardDesignResult:
        """Generate the next reward candidate using an OpenAI completion request."""

        if not self.openai_client.has_credentials:
            raise RuntimeError(
                "REWARDLAB_REWARD_DESIGN_MODE=openai requires OPENAI_API_KEY before reward "
                "iteration can use the model-backed designer"
            )

        allowed_parameter_names = request.allowed_parameter_names or _default_allowed_parameters(
            request.current_candidate
        )
        disallowed_reward_signatures = _candidate_reward_signatures(request.prior_candidates)
        base_messages: tuple[ChatMessage, ...] = (
            ChatMessage(
                role="system",
                content=(
                    "You are an RL reward engineer improving a Python reward "
                    "function for iterative Gymnasium experiments. Return JSON only."
                ),
            ),
            ChatMessage(
                role="user",
                content=_build_design_prompt(
                    request=request,
                    allowed_parameter_names=allowed_parameter_names,
                ),
            ),
        )

        max_attempts = 3
        last_error: RuntimeError | None = None
        last_content = ""
        for attempt in range(1, max_attempts + 1):
            messages: tuple[ChatMessage, ...] = base_messages
            if last_error is not None:
                messages = (
                    *messages,
                    ChatMessage(
                        role="user",
                        content=_build_retry_instruction(
                            error_message=str(last_error),
                            previous_content=last_content,
                            allowed_parameter_names=allowed_parameter_names,
                            disallowed_reward_signatures=tuple(
                                sorted(disallowed_reward_signatures)
                            ),
                        ),
                    ),
                )

            response = self.openai_client.chat_completion(
                ChatCompletionRequest(
                    model=self.config.model,
                    messages=messages,
                    reasoning_effort=self.config.reasoning_effort,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
            )

            try:
                return _parse_design_response(
                    response.content,
                    session_id=request.session_id,
                    iteration_index=request.next_iteration_index,
                    allowed_parameter_names=allowed_parameter_names,
                    disallowed_reward_signatures=disallowed_reward_signatures,
                    model_name=self.config.model,
                )
            except RuntimeError as exc:
                last_error = exc
                last_content = response.content
                if attempt >= max_attempts:
                    break

        assert last_error is not None
        raise RuntimeError(str(last_error)) from last_error


def resolve_reward_designer(
    *,
    config: RewardDesignConfig | None = None,
    openai_client: OpenAIClient | None = None,
) -> RewardDesigner:
    """Resolve the configured reward designer without triggering network activity."""

    resolved_config = config or RewardDesignConfig.from_environment()
    if resolved_config.mode == RewardDesignerMode.OPENAI:
        return OpenAIRewardDesigner(
            openai_client=openai_client,
            config=resolved_config,
        )
    return DeterministicRewardDesigner()


def _build_design_prompt(
    *,
    request: RewardDesignRequest,
    allowed_parameter_names: Sequence[str],
) -> str:
    """Build the prompt for one reward-improvement iteration."""

    prior_candidates = (
        list(request.prior_candidates)
        if len(request.prior_candidates) > 0
        else [request.current_candidate]
    )
    recent_candidates = sorted(
        prior_candidates,
        key=lambda candidate: candidate.iteration_index,
    )[-12:]
    candidate_history = [
        {
            "candidate_id": candidate.candidate_id,
            "iteration_index": candidate.iteration_index,
            "aggregate_score": candidate.aggregate_score,
            "change_summary": candidate.change_summary,
            "reward_signature": _reward_signature(candidate.reward_definition),
        }
        for candidate in recent_candidates
    ]
    disallowed_signatures = sorted(
        _candidate_reward_signatures(tuple(prior_candidates))
    )
    latest_metrics = (
        json.dumps(request.latest_run.metrics, sort_keys=True, indent=2)
        if request.latest_run is not None
        else "None yet. This is the zero-shot iteration from the baseline reward."
    )
    latest_reflection_summary = (
        request.latest_reflection.summary
        if request.latest_reflection is not None
        else "No prior reflection is available yet."
    )
    latest_proposed_changes = (
        "\n".join(f"- {item}" for item in request.latest_reflection.proposed_changes)
        if request.latest_reflection is not None
        else "- Start from the baseline reward and propose a stronger shaping signal."
    )
    decision_context = (
        json.dumps(list(request.recent_decisions)[-8:], indent=2)
        if len(request.recent_decisions) > 0
        else "None available."
    )
    feedback_context = (
        json.dumps(list(request.recent_feedback)[-6:], indent=2)
        if len(request.recent_feedback) > 0
        else "None available."
    )
    robustness_context = (
        json.dumps(list(request.recent_robustness_assessments)[-6:], indent=2)
        if len(request.recent_robustness_assessments) > 0
        else "None available."
    )
    signature_line = (
        ", ".join(disallowed_signatures[-16:])
        if len(disallowed_signatures) > 0
        else "none"
    )
    return (
        f"Session: {request.session_id}\n"
        f"Iteration to produce: {request.next_iteration_index}\n"
        f"Environment backend: {request.environment_backend.value}\n"
        f"Environment id: {request.environment_id}\n\n"
        f"Objective:\n{request.objective_text}\n\n"
        f"Current candidate score: {request.current_candidate.aggregate_score}\n"
        f"Current candidate change summary:\n{request.current_candidate.change_summary}\n\n"
        f"Current reward definition:\n{request.current_candidate.reward_definition}\n\n"
        f"Candidate history (recent):\n{json.dumps(candidate_history, indent=2)}\n\n"
        f"Recent decision context:\n{decision_context}\n\n"
        f"Recent human feedback context:\n{feedback_context}\n\n"
        f"Recent robustness context:\n{robustness_context}\n\n"
        f"Latest reflection summary:\n{latest_reflection_summary}\n\n"
        f"Latest proposed changes:\n{latest_proposed_changes}\n\n"
        f"Latest run metrics:\n{latest_metrics}\n\n"
        "Constraints:\n"
        "- Return a JSON object with keys reward_definition, change_summary, and "
        "proposed_changes.\n"
        "- reward_definition must be complete Python source code with a top-level callable "
        "named reward or compute_reward.\n"
        "- The callable must return a numeric scalar reward.\n"
        "- Do not wrap the code in Markdown fences.\n"
        "- Use only these allowed callable parameters: "
        f"{', '.join(allowed_parameter_names)}.\n"
        "- You may use a subset of those parameters, but do not require any other names.\n"
        "- Avoid cyclic revisions: do not repeat any previously used reward signature.\n"
        f"- Disallowed historical reward signatures: {signature_line}.\n"
        "- Keep the reward concise, executable, and directly aligned with the objective.\n"
    )


def _parse_design_response(
    content: str,
    *,
    session_id: str,
    iteration_index: int,
    allowed_parameter_names: Sequence[str],
    disallowed_reward_signatures: set[str],
    model_name: str,
) -> RewardDesignResult:
    """Parse, validate, and normalize a model-backed design response."""

    normalized_content = content.strip()
    if not normalized_content:
        raise RuntimeError("reward designer returned a blank response")

    try:
        payload = json.loads(normalized_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"reward designer returned invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("reward designer response must be a JSON object")

    reward_definition = _require_nonblank_string(payload, "reward_definition")
    change_summary = _require_nonblank_string(payload, "change_summary")
    proposed_changes = _require_nonempty_string_list(payload, "proposed_changes")

    preview_program = load_reward_program(
        candidate_id=f"{session_id}-preview-{iteration_index:03d}",
        source_text=reward_definition,
    )
    if preview_program.validation_status.value != "valid":
        raise RuntimeError(
            "reward designer produced invalid reward code: "
            f"{preview_program.validation_error or 'unknown validation error'}"
        )

    unsupported_parameters = sorted(
        _unsupported_parameters_for_signature(
            signature=inspect.signature(preview_program.require_callable()),
            allowed_parameter_names=allowed_parameter_names,
        )
    )
    if unsupported_parameters:
        joined = ", ".join(repr(name) for name in unsupported_parameters)
        raise RuntimeError(
            "reward designer introduced unsupported callable parameters: "
            f"{joined}"
        )
    proposed_signature = _reward_signature(reward_definition)
    if proposed_signature in disallowed_reward_signatures:
        raise RuntimeError(
            "reward designer produced a previously used reward definition "
            f"(signature={proposed_signature})"
        )

    return RewardDesignResult(
        reward_definition=reward_definition,
        change_summary=change_summary,
        proposed_changes=proposed_changes,
        model_name=model_name,
    )


def _require_nonblank_string(payload: dict[str, Any], key: str) -> str:
    """Return a non-blank string field from a model response payload."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"reward designer response is missing a non-blank {key!r} field")
    return value.strip()


def _require_nonempty_string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Return a validated list of non-blank strings from a model response payload."""

    value = payload.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"reward designer response field {key!r} must be a list")

    cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(cleaned) != len(value) or not cleaned:
        raise RuntimeError(
            f"reward designer response field {key!r} must contain non-blank strings"
        )
    return cleaned


def _unsupported_parameters_for_signature(
    *,
    signature: inspect.Signature,
    allowed_parameter_names: Sequence[str],
) -> list[str]:
    """Return unsupported fixed parameter names from a reward callable signature."""

    allowed = set(allowed_parameter_names)
    unsupported: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.name not in allowed:
            unsupported.append(parameter.name)
    return unsupported


def _default_allowed_parameters(candidate: RewardCandidate) -> tuple[str, ...]:
    """Return the safe callable parameters allowed for iterative reward revisions."""

    names = list(GENERIC_REWARD_PARAMETER_NAMES)
    program = load_reward_program(
        candidate_id=candidate.candidate_id,
        source_text=candidate.reward_definition,
    )
    if program.validation_status.value == "valid":
        names.extend(program.parameter_names())
    return tuple(dict.fromkeys(names))


def _build_retry_instruction(
    *,
    error_message: str,
    previous_content: str,
    allowed_parameter_names: Sequence[str],
    disallowed_reward_signatures: Sequence[str],
) -> str:
    """Build a repair prompt after an invalid model response."""

    previous_excerpt = previous_content.strip() or "<blank>"
    disallowed = (
        ", ".join(disallowed_reward_signatures[-16:])
        if disallowed_reward_signatures
        else "none"
    )
    return (
        "Your previous JSON was invalid for execution.\n"
        f"Validation error: {error_message}\n"
        f"Previous response: {previous_excerpt}\n\n"
        "Return corrected JSON only with keys reward_definition, change_summary, and "
        "proposed_changes.\n"
        "- Do not use *args or **kwargs in the reward signature.\n"
        "- Use only allowed callable parameters: "
        f"{', '.join(allowed_parameter_names)}.\n"
        "- Do not repeat these historical reward signatures: "
        f"{disallowed}.\n"
        "- Keep the reward executable Python source without Markdown fences.\n"
    )


def _candidate_reward_signatures(candidates: Sequence[RewardCandidate]) -> set[str]:
    """Return signatures for previously used reward definitions."""

    return {_reward_signature(candidate.reward_definition) for candidate in candidates}


def _reward_signature(source_text: str) -> str:
    """Return a stable signature for reward code to detect cyclical repeats."""

    normalized_lines = []
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        normalized_lines.append("".join(stripped.split()))
    normalized_source = "\n".join(normalized_lines).strip().lower()
    digest = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()
    return digest[:12]


def _reasoning_effort_from_environment(
    raw_value: str | None,
) -> Literal["minimal", "low", "medium", "high"]:
    """Parse the configured reasoning effort for reward generation."""

    if raw_value is None or raw_value == "":
        return "medium"
    normalized = raw_value.strip().lower()
    if normalized not in {"minimal", "low", "medium", "high"}:
        raise ValueError(
            "REWARDLAB_REWARD_DESIGN_REASONING_EFFORT must be one of "
            "'minimal', 'low', 'medium', or 'high'"
        )
    return cast(Literal["minimal", "low", "medium", "high"], normalized)


def _int_from_environment(raw_value: str | None, *, default: int) -> int:
    """Parse a positive integer value from the environment with a fallback."""

    if raw_value is None or raw_value == "":
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return max(parsed, 1)


def _revise_reward_definition(
    current_definition: str,
    *,
    objective_text: str,
    proposed_changes: list[str],
    iteration_index: int,
) -> str:
    """Return a revised reward definition with deterministic improvement notes."""

    objective_terms = ", ".join(sorted(set(_tokenize(objective_text)))[:6])
    change_block = "\n".join(f"# - {change}" for change in proposed_changes)
    return (
        f"{current_definition.rstrip()}\n\n"
        f"# Iteration {iteration_index} refinement\n"
        f"# Objective terms: {objective_terms}\n"
        f"{change_block}\n"
    )


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized alphanumeric terms."""

    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        if character.isalnum() or character == "_":
            current.append(character.lower())
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
