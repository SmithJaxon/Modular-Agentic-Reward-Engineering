"""
Summary: Reward program loading and validation helpers for real backend execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from rewardlab.utils.compat import StrEnum
from types import CodeType
from typing import Any

from rewardlab.schemas.reward_candidate import RewardCandidate

DEFAULT_ENTRYPOINT_NAME = "reward"
LEGACY_ENTRYPOINT_NAME = "compute_reward"
SIGNATURE_VERSION = "rewardlab.reward-program.v1"

__all__ = [
    "DEFAULT_ENTRYPOINT_NAME",
    "LEGACY_ENTRYPOINT_NAME",
    "RewardProgram",
    "RewardProgramStatus",
    "SIGNATURE_VERSION",
    "load_reward_program",
    "load_reward_program_from_candidate",
]


class RewardProgramStatus(StrEnum):
    """Validation state for a reward program loaded from candidate source."""

    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RewardProgram:
    """Validated reward-program source plus the resolved runtime callable."""

    candidate_id: str
    entrypoint_name: str
    source_text: str
    signature_version: str
    validation_status: RewardProgramStatus
    validation_error: str | None = None
    reward_callable: Any | None = field(default=None, repr=False, compare=False)

    def require_callable(self) -> Any:
        """Return the loaded reward callable or raise an actionable validation error."""

        if self.validation_status != RewardProgramStatus.VALID or self.reward_callable is None:
            raise ValueError(self.validation_error or "reward program is not executable")
        return self.reward_callable

    def parameter_names(self) -> tuple[str, ...]:
        """Expose the resolved callable signature for later runtime adaptation."""

        return tuple(inspect.signature(self.require_callable()).parameters)


def load_reward_program_from_candidate(
    candidate: RewardCandidate,
    *,
    entrypoint_name: str = DEFAULT_ENTRYPOINT_NAME,
    runtime_compat_profile: str | None = None,
) -> RewardProgram:
    """Load a reward program from a stored reward candidate definition."""

    return load_reward_program(
        candidate_id=candidate.candidate_id,
        source_text=candidate.reward_definition,
        entrypoint_name=entrypoint_name,
        runtime_compat_profile=runtime_compat_profile,
    )


def load_reward_program(
    *,
    candidate_id: str,
    source_text: str,
    entrypoint_name: str = DEFAULT_ENTRYPOINT_NAME,
    runtime_compat_profile: str | None = None,
) -> RewardProgram:
    """Compile candidate source and resolve a callable reward entrypoint."""

    normalized_source = source_text.strip()
    if not normalized_source:
        return _invalid_program(
            candidate_id,
            entrypoint_name,
            source_text,
            "reward source is blank",
        )
    normalized_source = _apply_runtime_compat_rewrites(
        normalized_source,
        runtime_compat_profile=runtime_compat_profile,
    )

    try:
        compiled = compile(
            normalized_source,
            filename=f"<reward-program:{candidate_id}>",
            mode="exec",
        )
    except SyntaxError as exc:
        return _invalid_program(
            candidate_id,
            entrypoint_name,
            source_text,
            f"reward source has invalid Python syntax at line {exc.lineno}: {exc.msg}",
        )

    try:
        namespace = _execute_compiled_program(candidate_id, compiled)
    except Exception as exc:
        return _invalid_program(
            candidate_id,
            entrypoint_name,
            source_text,
            f"reward source raised during import: {exc}",
        )

    resolved_entrypoint = _resolve_entrypoint_name(namespace, entrypoint_name)
    if resolved_entrypoint is None:
        looked_for = ", ".join(f"'{name}'" for name in _candidate_entrypoint_names(entrypoint_name))
        return _invalid_program(
            candidate_id,
            entrypoint_name,
            source_text,
            f"reward program must define a callable entrypoint; looked for {looked_for}",
        )

    reward_callable = namespace[resolved_entrypoint]
    if not callable(reward_callable):
        return _invalid_program(
            candidate_id,
            resolved_entrypoint,
            source_text,
            f"reward entrypoint '{resolved_entrypoint}' is not callable",
        )

    return RewardProgram(
        candidate_id=candidate_id,
        entrypoint_name=resolved_entrypoint,
        source_text=source_text,
        signature_version=SIGNATURE_VERSION,
        validation_status=RewardProgramStatus.VALID,
        reward_callable=reward_callable,
    )


def _execute_compiled_program(candidate_id: str, compiled: CodeType) -> dict[str, Any]:
    """Execute compiled reward source in an isolated module-like namespace."""

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": f"rewardlab_reward_{candidate_id.replace('-', '_')}",
    }
    exec(compiled, namespace, namespace)  # noqa: S102 - constrained reward loader boundary
    return namespace


def _apply_runtime_compat_rewrites(
    source_text: str,
    *,
    runtime_compat_profile: str | None,
) -> str:
    """Apply backend-specific source rewrites needed for runtime compatibility."""

    if runtime_compat_profile != "isaacgym":
        return source_text
    return _rewrite_torch_compile_calls(source_text)


def _rewrite_torch_compile_calls(source_text: str) -> str:
    """Rewrite torch.compile usage to a runtime-safe helper for Torch 1.x."""

    if "torch.compile(" not in source_text and "@torch.compile" not in source_text:
        return source_text
    shim = (
        "def _rewardlab_torch_compile(fn, *args, **kwargs):\n"
        "    try:\n"
        "        compiler = torch.compile\n"
        "    except Exception:\n"
        "        return fn\n"
        "    return compiler(fn, *args, **kwargs)\n\n"
    )
    rewritten = source_text.replace("torch.compile(", "_rewardlab_torch_compile(")
    rewritten = rewritten.replace("@torch.compile", "@_rewardlab_torch_compile")
    return shim + rewritten


def _resolve_entrypoint_name(namespace: dict[str, Any], requested_name: str) -> str | None:
    """Resolve the requested or legacy entrypoint name from the executed namespace."""

    for candidate_name in _candidate_entrypoint_names(requested_name):
        if callable(namespace.get(candidate_name)):
            return candidate_name
    return None


def _candidate_entrypoint_names(requested_name: str) -> tuple[str, ...]:
    """Return the accepted entrypoint names in lookup priority order."""

    if requested_name == DEFAULT_ENTRYPOINT_NAME:
        return (DEFAULT_ENTRYPOINT_NAME, LEGACY_ENTRYPOINT_NAME)
    return (requested_name,)


def _invalid_program(
    candidate_id: str,
    entrypoint_name: str,
    source_text: str,
    validation_error: str,
) -> RewardProgram:
    """Construct an invalid reward program with the supplied validation error."""

    return RewardProgram(
        candidate_id=candidate_id,
        entrypoint_name=entrypoint_name,
        source_text=source_text,
        signature_version=SIGNATURE_VERSION,
        validation_status=RewardProgramStatus.INVALID,
        validation_error=validation_error,
    )

