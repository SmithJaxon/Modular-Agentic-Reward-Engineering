"""
Summary: Session lifecycle orchestration service for start, step, stop, and resume flows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rewardlab.orchestrator.checkpointing import build_checkpoint_payload
from rewardlab.orchestrator.iteration_engine import IterationEngine
from rewardlab.orchestrator.reporting import build_report_payload, write_report
from rewardlab.orchestrator.state_machine import (
    SessionLifecycleState,
    ensure_transition,
)
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.session_config import SessionConfig


class SessionService:
    """
    Coordinate session lifecycle operations across repository and iteration logic.
    """

    def __init__(
        self,
        repository: SessionRepository,
        iteration_engine: IterationEngine | None = None,
    ) -> None:
        """
        Initialize session service dependencies.

        Args:
            repository: Session repository facade.
            iteration_engine: Optional iteration engine override.
        """
        self._repository = repository
        self._iteration_engine = iteration_engine or IterationEngine()

    def start_session(
        self,
        config: SessionConfig,
        baseline_reward_definition: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new running session and initial checkpoint state.

        Args:
            config: Validated session configuration.
            baseline_reward_definition: Baseline reward text from input file.
            session_id: Optional caller-provided session identifier.

        Returns:
            Created session metadata.
        """
        created = self._repository.create_session(config=config, session_id=session_id)
        metadata = {
            **created.get("metadata", {}),
            "baseline_reward_definition": baseline_reward_definition.strip(),
            "iteration_index": -1,
            "no_improve_streak": 0,
            "last_score": None,
        }
        self._repository.update_session(created["session_id"], metadata=metadata)
        refreshed = self._require_session(created["session_id"])
        checkpoint_payload = build_checkpoint_payload(refreshed, [], None)
        self._repository.checkpoint(refreshed["session_id"], checkpoint_payload)
        return refreshed

    def step_session(self, session_id: str) -> dict[str, Any]:
        """
        Execute one full iteration and update session state.

        Args:
            session_id: Session identifier.

        Returns:
            Step summary payload for CLI responses.
        """
        session = self._require_session(session_id)
        current_state = SessionLifecycleState(session["status"])
        if current_state != SessionLifecycleState.RUNNING:
            raise RuntimeError(f"session {session_id} is not running")

        metadata = dict(session.get("metadata", {}))
        baseline_reward = metadata.get("baseline_reward_definition", "reward = 0")
        iteration_index = int(metadata.get("iteration_index", -1)) + 1

        result = self._iteration_engine.run_iteration(
            session=session,
            iteration_index=iteration_index,
            baseline_reward_definition=baseline_reward,
        )
        candidate = self._repository.add_candidate(
            session_id=session_id,
            iteration_index=iteration_index,
            reward_definition=result.reward_definition,
            change_summary=result.change_summary,
            aggregate_score=result.score,
        )
        reflection = self._repository.add_reflection(
            candidate_id=candidate["candidate_id"],
            summary=result.reflection_summary,
            proposed_changes=result.proposed_changes,
            confidence=result.confidence,
        )

        previous_best = self._repository.get_best_candidate(session_id)
        previous_best_score = previous_best["aggregate_score"] if previous_best else None
        improved = previous_best_score is None or result.score > previous_best_score
        no_improve_streak = 0 if improved else (int(metadata.get("no_improve_streak", 0)) + 1)

        candidates = self._repository.list_candidates(session_id)
        best_candidate = max(candidates, key=lambda item: item["aggregate_score"])
        self._repository.set_best_candidate(session_id, best_candidate["candidate_id"])

        next_status = SessionLifecycleState.RUNNING
        next_stop_reason: str | None = None
        if no_improve_streak >= session["no_improve_limit"]:
            next_status = SessionLifecycleState.COMPLETED
            next_stop_reason = "convergence"
        elif iteration_index + 1 >= session["max_iterations"]:
            next_status = SessionLifecycleState.COMPLETED
            next_stop_reason = "iteration_cap"

        updated_metadata = {
            **metadata,
            "iteration_index": iteration_index,
            "no_improve_streak": no_improve_streak,
            "last_score": result.score,
        }
        self._repository.update_session(
            session_id,
            metadata=updated_metadata,
            status=next_status.value,
            stop_reason=next_stop_reason,
        )
        updated_session = self._require_session(session_id)
        checkpoint_payload = build_checkpoint_payload(updated_session, candidates, reflection)
        self._repository.checkpoint(session_id, checkpoint_payload)
        return {
            "session_id": session_id,
            "iteration_index": iteration_index,
            "candidate_id": candidate["candidate_id"],
            "status": updated_session["status"],
            "best_candidate_id": updated_session["best_candidate_id"],
            "performance_summary": result.performance_summary,
        }

    def pause_session(self, session_id: str) -> dict[str, Any]:
        """
        Pause a running session and persist checkpoint metadata.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session metadata.
        """
        session = self._require_session(session_id)
        ensure_transition(
            SessionLifecycleState(session["status"]),
            SessionLifecycleState.PAUSED,
        )
        self._repository.update_session(session_id, status=SessionLifecycleState.PAUSED.value)
        paused = self._require_session(session_id)
        candidates = self._repository.list_candidates(session_id)
        self._repository.checkpoint(session_id, build_checkpoint_payload(paused, candidates, None))
        return paused

    def resume_session(self, session_id: str) -> dict[str, Any]:
        """
        Resume a paused session.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session metadata.
        """
        session = self._require_session(session_id)
        ensure_transition(
            SessionLifecycleState(session["status"]),
            SessionLifecycleState.RUNNING,
        )
        self._repository.update_session(session_id, status=SessionLifecycleState.RUNNING.value)
        return self._require_session(session_id)

    def stop_session(self, session_id: str, report_dir: Path) -> dict[str, Any]:
        """
        Interrupt a running session and export current best-candidate report.

        Args:
            session_id: Session identifier.
            report_dir: Destination directory for report artifacts.

        Returns:
            Stop response payload.
        """
        session = self._require_session(session_id)
        if SessionLifecycleState(session["status"]) != SessionLifecycleState.RUNNING:
            raise RuntimeError("session must be running before stop")
        ensure_transition(
            SessionLifecycleState(session["status"]),
            SessionLifecycleState.INTERRUPTED,
        )
        self._repository.update_session(
            session_id,
            status=SessionLifecycleState.INTERRUPTED.value,
            stop_reason="user_interrupt",
        )
        updated = self._require_session(session_id)
        candidates = self._repository.list_candidates(session_id)
        if not candidates:
            placeholder = self._repository.add_candidate(
                session_id=session_id,
                iteration_index=0,
                reward_definition=updated["metadata"].get(
                    "baseline_reward_definition",
                    "reward = 0",
                ),
                change_summary="No iteration executed before stop",
                aggregate_score=0.0,
            )
            self._repository.set_best_candidate(session_id, placeholder["candidate_id"])
            updated = self._require_session(session_id)
            candidates = self._repository.list_candidates(session_id)
        report = build_report_payload(updated, candidates)
        path = write_report(report, report_dir)
        self._repository.checkpoint(session_id, build_checkpoint_payload(updated, candidates, None))
        return {
            "session_id": session_id,
            "stop_reason": "user_interrupt",
            "best_candidate_id": report.best_candidate.candidate_id,
            "report_path": str(path),
        }

    def report_session(self, session_id: str, report_dir: Path) -> dict[str, Any]:
        """
        Export a report for a session without changing lifecycle state.

        Args:
            session_id: Session identifier.
            report_dir: Destination directory for report artifacts.

        Returns:
            Report summary payload.
        """
        session = self._require_session(session_id)
        candidates = self._repository.list_candidates(session_id)
        report = build_report_payload(session, candidates)
        path = write_report(report, report_dir)
        return {"session_id": session_id, "report_path": str(path)}

    def _require_session(self, session_id: str) -> dict[str, Any]:
        """
        Fetch session and raise explicit error when missing.

        Args:
            session_id: Session identifier.

        Returns:
            Session metadata dictionary.
        """
        session = self._repository.get_session(session_id)
        if session is None:
            raise RuntimeError(f"session {session_id} not found")
        return session
