"""
Summary: Worker tool that validates a candidate reward program for executability.
Created: 2026-04-16
Last Updated: 2026-04-16
"""

from __future__ import annotations

from rewardlab.agentic.contracts import ToolResult
from rewardlab.experiments.reward_program import load_reward_program_from_candidate
from rewardlab.schemas.agent_experiment import AgentExperimentRecord
from rewardlab.schemas.experiment_run import ExperimentRun
from rewardlab.schemas.reward_candidate import RewardCandidate


class ValidateRewardProgramTool:
    """Validate reward source for a selected candidate before execution."""

    def execute(
        self,
        *,
        record: AgentExperimentRecord,
        candidates: list[RewardCandidate],
        runs: list[ExperimentRun],
        action_input: dict[str, object],
    ) -> ToolResult:
        """Return validation status and callable signature details."""

        del runs
        selected_candidate = _select_candidate(candidates=candidates, action_input=action_input)
        entrypoint_name = record.spec.baseline_reward.entrypoint_name
        program = load_reward_program_from_candidate(
            selected_candidate,
            entrypoint_name=entrypoint_name,
        )

        if program.validation_status.value != "valid":
            error_message = program.validation_error or "reward program is invalid"
            return ToolResult(
                status="error",
                summary=(
                    f"Validation failed for {selected_candidate.candidate_id}: {error_message}"
                ),
                payload={
                    "candidate_id": selected_candidate.candidate_id,
                    "entrypoint_name": program.entrypoint_name,
                    "validation_status": program.validation_status.value,
                    "validation_error": error_message,
                },
            )

        parameter_names = list(program.parameter_names())
        return ToolResult(
            status="ok",
            summary=(
                f"Validated {selected_candidate.candidate_id} "
                f"with entrypoint {program.entrypoint_name}."
            ),
            payload={
                "candidate_id": selected_candidate.candidate_id,
                "entrypoint_name": program.entrypoint_name,
                "validation_status": program.validation_status.value,
                "parameter_names": parameter_names,
                "signature_version": program.signature_version,
            },
        )


def _select_candidate(
    *,
    candidates: list[RewardCandidate],
    action_input: dict[str, object],
) -> RewardCandidate:
    """Return candidate requested by action input or fallback to latest iteration."""

    requested_id = action_input.get("candidate_id")
    if isinstance(requested_id, str):
        for candidate in candidates:
            if candidate.candidate_id == requested_id:
                return candidate
    return max(candidates, key=lambda candidate: candidate.iteration_index)
