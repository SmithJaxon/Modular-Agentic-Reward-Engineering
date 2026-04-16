"""
Summary: Session lifecycle orchestration service for session, feedback, and reporting flows.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from rewardlab.experiments.backends.base import ExperimentInput
from rewardlab.experiments.robustness_runner import RobustnessRunner
from rewardlab.feedback.gating import (
    CandidateFeedbackState,
    evaluate_feedback_gate,
    summarize_feedback_by_candidate,
)
from rewardlab.feedback.human_feedback_service import HumanFeedbackService
from rewardlab.feedback.peer_feedback_client import PeerFeedbackClient, PeerReviewContext
from rewardlab.orchestrator.budget_manager import (
    budget_exhausted_for_next_iteration,
    initialize_budget_metadata,
    plan_iteration_budget,
    record_iteration_budget_usage,
    remaining_budget_snapshot,
)
from rewardlab.orchestrator.checkpointing import build_checkpoint_payload
from rewardlab.orchestrator.iteration_engine import IterationEngine
from rewardlab.orchestrator.reporting import build_report_payload, write_report
from rewardlab.orchestrator.state_machine import (
    SessionLifecycleState,
    ensure_transition,
)
from rewardlab.persistence.session_repository import SessionRepository
from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSource
from rewardlab.schemas.robustness_assessment import RiskLevel
from rewardlab.schemas.session_config import EnvironmentBackend, FeedbackGate, SessionConfig
from rewardlab.selection.policy import CandidateSignal, SelectionOutcome, select_candidate


class SessionService:
    """
    Coordinate session lifecycle operations across repository and iteration logic.
    """

    def __init__(
        self,
        repository: SessionRepository,
        iteration_engine: IterationEngine | None = None,
        robustness_runner: RobustnessRunner | None = None,
        human_feedback_service: HumanFeedbackService | None = None,
        peer_feedback_client: PeerFeedbackClient | None = None,
    ) -> None:
        """
        Initialize session service dependencies.

        Args:
            repository: Session repository facade.
            iteration_engine: Optional iteration engine override.
            robustness_runner: Optional robustness runner override.
            human_feedback_service: Optional human feedback service override.
            peer_feedback_client: Optional peer feedback client override.
        """
        self._repository = repository
        self._iteration_engine = iteration_engine or IterationEngine()
        self._robustness_runner = robustness_runner or RobustnessRunner()
        self._human_feedback_service = human_feedback_service or HumanFeedbackService(
            repository=repository
        )
        self._peer_feedback_client = peer_feedback_client or PeerFeedbackClient()

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
        metadata: dict[str, Any] = {
            **created.get("metadata", {}),
            "artifact_root": str(self._repository.root_dir),
            "baseline_reward_definition": baseline_reward_definition.strip(),
            "iteration_index": -1,
            "no_improve_streak": 0,
            "last_score": None,
            "candidate_primary_scores": {},
            "candidate_performance_summaries": {},
            "robustness_assessments": {},
            "candidate_feedback_counts": {},
            "candidate_feedback_summaries": {},
            "candidate_feedback_bonuses": {},
            "eligible_best_candidate_id": None,
            "eligible_selection_summary": None,
            "eligible_minor_robustness_risk_accepted": False,
            "pending_feedback_summary": None,
            "demo_artifacts": {},
            "selected_minor_robustness_risk_accepted": False,
        }
        metadata = initialize_budget_metadata(created, metadata)
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
        seed_candidate = self._repository.get_best_candidate(session_id)
        seed_reward_definition = (
            seed_candidate["reward_definition"] if seed_candidate is not None else baseline_reward
        )
        seed_reflection = (
            self._repository.get_latest_reflection_for_candidate(seed_candidate["candidate_id"])
            if seed_candidate is not None
            else None
        )
        seed_reflection_summary = str(seed_reflection["summary"]) if seed_reflection else ""
        iteration_index = int(metadata.get("iteration_index", -1)) + 1
        pending_feedback_summary = str(metadata.get("pending_feedback_summary") or "").strip()
        budget_plan = plan_iteration_budget(session, metadata, iteration_index)
        runtime_metadata = dict(metadata)
        if budget_plan is not None:
            runtime_metadata.update(budget_plan.as_metadata_overrides())
        runtime_session = dict(session)
        runtime_session["metadata"] = runtime_metadata

        run_iteration_parameters = inspect.signature(
            self._iteration_engine.run_iteration
        ).parameters
        run_iteration_kwargs: dict[str, Any] = {
            "session": runtime_session,
            "iteration_index": iteration_index,
            "baseline_reward_definition": seed_reward_definition,
        }
        if "feedback_summary" in run_iteration_parameters:
            run_iteration_kwargs["feedback_summary"] = pending_feedback_summary
        if "seed_reflection_summary" in run_iteration_parameters:
            run_iteration_kwargs["seed_reflection_summary"] = seed_reflection_summary
        result = self._iteration_engine.run_iteration(**run_iteration_kwargs)
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

        robustness = self._robustness_runner.run(
            candidate_id=candidate["candidate_id"],
            payload=ExperimentInput(
                session_id=session["session_id"],
                environment_id=session["environment_id"],
                environment_backend=EnvironmentBackend(session["environment_backend"]),
                reward_definition=result.reward_definition,
                iteration_index=iteration_index,
                objective_text=session["objective_text"],
                overrides=dict(result.performance_metrics.get("overrides", {})),
            ),
            primary_score=result.score,
        )
        current_signal = CandidateSignal(
            candidate_id=candidate["candidate_id"],
            primary_performance=result.score,
            robustness_bonus=robustness.analysis.robustness_bonus,
            risk_level=robustness.assessment.risk_level,
            tradeoff_rationale=robustness.analysis.tradeoff_rationale,
        )
        self._repository.update_candidate(
            candidate["candidate_id"],
            aggregate_score=current_signal.aggregate_score,
        )
        candidate["aggregate_score"] = current_signal.aggregate_score

        improved = (
            previous_best_score is None or current_signal.aggregate_score > previous_best_score
        )
        no_improve_streak = 0 if improved else (int(metadata.get("no_improve_streak", 0)) + 1)

        candidate_primary_scores = dict(metadata.get("candidate_primary_scores", {}))
        candidate_primary_scores[candidate["candidate_id"]] = result.score
        performance_summaries = dict(metadata.get("candidate_performance_summaries", {}))
        performance_summaries[candidate["candidate_id"]] = result.performance_summary
        robustness_assessments = dict(metadata.get("robustness_assessments", {}))
        robustness_assessments[candidate["candidate_id"]] = robustness.assessment.model_dump()
        updated_metadata = record_iteration_budget_usage(
            metadata,
            candidate_id=candidate["candidate_id"],
            iteration_index=iteration_index,
            performance_metrics=result.performance_metrics,
            robustness_runs=list(robustness.experiment_runs),
            llm_calls_used=result.llm_calls_used,
            plan=budget_plan,
        )

        feedback_states = self._list_feedback_states(session_id)
        candidates = self._repository.list_candidates(session_id)
        signals = self._build_candidate_signals(
            candidates=candidates,
            primary_scores=candidate_primary_scores,
            assessments=robustness_assessments,
            feedback_states=feedback_states,
        )
        selection = select_candidate(signals)
        eligible_selection = self._select_eligible_candidate(
            gate=FeedbackGate(session["feedback_gate"]),
            signals=signals,
            feedback_states=feedback_states,
        )
        self._repository.set_best_candidate(session_id, selection.selected_signal.candidate_id)

        next_status = SessionLifecycleState.RUNNING
        next_stop_reason: str | None = None
        if no_improve_streak >= session["no_improve_limit"]:
            next_status = SessionLifecycleState.COMPLETED
            next_stop_reason = "convergence"
        elif iteration_index + 1 >= session["max_iterations"]:
            next_status = SessionLifecycleState.COMPLETED
            next_stop_reason = "iteration_cap"
        elif budget_exhausted_for_next_iteration(
            session,
            updated_metadata,
            next_iteration_index=iteration_index + 1,
        ):
            next_status = SessionLifecycleState.COMPLETED
            next_stop_reason = "budget_cap"

        updated_metadata = {
            **updated_metadata,
            "iteration_index": iteration_index,
            "no_improve_streak": no_improve_streak,
            "last_score": current_signal.aggregate_score,
            "candidate_primary_scores": candidate_primary_scores,
            "candidate_performance_summaries": performance_summaries,
            "robustness_assessments": robustness_assessments,
            "candidate_feedback_counts": {
                key: value.feedback_count for key, value in feedback_states.items()
            },
            "candidate_feedback_summaries": {
                key: value.summary for key, value in feedback_states.items()
            },
            "candidate_feedback_bonuses": {
                key: {
                    "human_feedback_bonus": value.human_feedback_bonus,
                    "peer_feedback_bonus": value.peer_feedback_bonus,
                }
                for key, value in feedback_states.items()
            },
            "eligible_best_candidate_id": (
                eligible_selection.selected_signal.candidate_id if eligible_selection else None
            ),
            "eligible_selection_summary": (
                eligible_selection.selection_summary if eligible_selection else None
            ),
            "eligible_minor_robustness_risk_accepted": (
                eligible_selection.selected_signal.minor_robustness_risk_accepted
                if eligible_selection
                else False
            ),
            "pending_feedback_summary": None,
            "selection_summary": selection.selection_summary,
            "selected_minor_robustness_risk_accepted": (
                selection.selected_signal.minor_robustness_risk_accepted
            ),
        }
        if selection.selected_signal.tradeoff_rationale is not None:
            updated_metadata["selected_tradeoff_rationale"] = (
                selection.selected_signal.tradeoff_rationale
            )
        else:
            updated_metadata.pop("selected_tradeoff_rationale", None)
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
            "feedback_summary": result.feedback_summary,
            "remaining_budget": remaining_budget_snapshot(
                updated_session,
                dict(updated_session.get("metadata", {})),
            ),
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

    def submit_human_feedback(
        self,
        session_id: str,
        candidate_id: str,
        comment: str,
        score: float | None = None,
        artifact_ref: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist human feedback and refresh gate-aware recommendation metadata.

        Args:
            session_id: Session identifier.
            candidate_id: Candidate identifier.
            comment: Reviewer comment text.
            score: Optional normalized score delta.
            artifact_ref: Optional demonstration artifact reference.

        Returns:
            Persisted feedback metadata dictionary.
        """
        session = self._require_session(session_id)
        candidate = self._require_candidate(candidate_id)
        payload = self._human_feedback_service.submit_feedback(
            session=session,
            candidate=candidate,
            comment=comment,
            score=score,
            artifact_ref=artifact_ref,
        )
        self._refresh_feedback_state(session_id, pending_candidate_id=candidate_id)
        return payload

    def request_peer_feedback(self, session_id: str, candidate_id: str) -> dict[str, Any]:
        """
        Generate deterministic peer feedback from isolated candidate context.

        Args:
            session_id: Session identifier.
            candidate_id: Candidate identifier.

        Returns:
            Persisted peer feedback metadata dictionary.
        """
        session = self._require_session(session_id)
        candidate = self._require_candidate(candidate_id)
        if candidate["session_id"] != session_id:
            raise RuntimeError(f"candidate {candidate_id} does not belong to session {session_id}")
        metadata = dict(session.get("metadata", {}))
        context = PeerReviewContext(
            candidate_id=candidate_id,
            iteration_index=int(candidate["iteration_index"]),
            objective_text=session["objective_text"],
            environment_backend=EnvironmentBackend(session["environment_backend"]),
            performance_summary=str(
                dict(metadata.get("candidate_performance_summaries", {})).get(
                    candidate_id,
                    (
                        f"iteration {candidate['iteration_index']} "
                        f"score={candidate['aggregate_score']:.3f}"
                    ),
                )
            ),
            risk_level=RiskLevel(
                dict(metadata.get("robustness_assessments", {}))
                .get(candidate_id, {})
                .get("risk_level", RiskLevel.LOW.value)
            ),
            artifact_refs=tuple(
                str(value)
                for value in dict(metadata.get("demo_artifacts", {})).get(candidate_id, [])
            ),
        )
        draft = self._peer_feedback_client.request_feedback(context)
        payload = self._repository.add_feedback(
            candidate_id=candidate_id,
            source_type=FeedbackSource.PEER.value,
            comment=draft.comment,
            score=draft.score,
            artifact_ref=draft.artifact_ref,
        )
        FeedbackEntry.model_validate(payload)
        self._refresh_feedback_state(session_id, pending_candidate_id=candidate_id)
        return payload

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

    def _require_candidate(self, candidate_id: str) -> dict[str, Any]:
        """
        Fetch candidate and raise explicit error when missing.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            Candidate metadata dictionary.
        """
        candidate = self._repository.get_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError(f"candidate {candidate_id} not found")
        return candidate

    @staticmethod
    def _build_candidate_signals(
        candidates: list[dict[str, Any]],
        primary_scores: dict[str, float],
        assessments: dict[str, dict[str, Any]],
        feedback_states: dict[str, CandidateFeedbackState],
    ) -> list[CandidateSignal]:
        """
        Rebuild policy inputs from stored candidate and robustness metadata.

        Args:
            candidates: Persisted candidate rows.
            primary_scores: Primary score mapping per candidate.
            assessments: Robustness assessment mapping per candidate.
            feedback_states: Aggregated feedback state mapping per candidate.

        Returns:
            Candidate signal values ready for policy selection.
        """
        signals: list[CandidateSignal] = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            primary_score = float(primary_scores.get(candidate_id, candidate["aggregate_score"]))
            assessment = assessments.get(candidate_id, {})
            feedback_state = feedback_states.get(
                candidate_id,
                CandidateFeedbackState(candidate_id=candidate_id),
            )
            risk_level_value = str(assessment.get("risk_level", RiskLevel.LOW.value))
            tradeoff_rationale = None
            if risk_level_value == RiskLevel.MEDIUM.value:
                tradeoff_rationale = (
                    "primary performance remained strong while probe degradation stayed "
                    "within discretionary bounds"
                )
            signals.append(
                CandidateSignal(
                    candidate_id=candidate_id,
                    primary_performance=primary_score,
                    robustness_bonus=float(candidate["aggregate_score"]) - primary_score,
                    human_feedback_bonus=feedback_state.human_feedback_bonus,
                    peer_feedback_bonus=feedback_state.peer_feedback_bonus,
                    risk_level=RiskLevel(risk_level_value),
                    tradeoff_rationale=tradeoff_rationale,
                )
            )
        return signals

    def _list_feedback_states(self, session_id: str) -> dict[str, CandidateFeedbackState]:
        """
        Load and summarize persisted feedback entries for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Mapping of candidate identifier to aggregated feedback state.
        """
        entries = [
            FeedbackEntry.model_validate(payload)
            for payload in self._repository.list_feedback_for_session(session_id)
        ]
        return summarize_feedback_by_candidate(entries)

    @staticmethod
    def _select_eligible_candidate(
        gate: FeedbackGate,
        signals: list[CandidateSignal],
        feedback_states: dict[str, CandidateFeedbackState],
    ) -> SelectionOutcome | None:
        """
        Select the best candidate among those satisfying the configured feedback gate.

        Args:
            gate: Session feedback gate configuration.
            signals: Policy candidate signal values.
            feedback_states: Aggregated feedback state mapping per candidate.

        Returns:
            Selection outcome for eligible candidates or None when none qualify.
        """
        eligible = [
            signal
            for signal in signals
            if evaluate_feedback_gate(
                gate,
                feedback_states.get(
                    signal.candidate_id,
                    CandidateFeedbackState(candidate_id=signal.candidate_id),
                ),
            ).satisfied
        ]
        if not eligible:
            return None
        return select_candidate(eligible)

    def _refresh_feedback_state(
        self,
        session_id: str,
        pending_candidate_id: str | None = None,
    ) -> None:
        """
        Recompute feedback summaries and gate-aware recommendation metadata.

        Args:
            session_id: Session identifier.
            pending_candidate_id: Candidate whose feedback should inform the next step.
        """
        session = self._require_session(session_id)
        metadata = dict(session.get("metadata", {}))
        feedback_states = self._list_feedback_states(session_id)
        candidates = self._repository.list_candidates(session_id)
        metadata["candidate_feedback_counts"] = {
            key: value.feedback_count for key, value in feedback_states.items()
        }
        metadata["candidate_feedback_summaries"] = {
            key: value.summary for key, value in feedback_states.items()
        }
        metadata["candidate_feedback_bonuses"] = {
            key: {
                "human_feedback_bonus": value.human_feedback_bonus,
                "peer_feedback_bonus": value.peer_feedback_bonus,
            }
            for key, value in feedback_states.items()
        }
        if pending_candidate_id is not None:
            pending_state = feedback_states.get(
                pending_candidate_id,
                CandidateFeedbackState(candidate_id=pending_candidate_id),
            )
            metadata["pending_feedback_summary"] = pending_state.summary

        if candidates:
            signals = self._build_candidate_signals(
                candidates=candidates,
                primary_scores=dict(metadata.get("candidate_primary_scores", {})),
                assessments=dict(metadata.get("robustness_assessments", {})),
                feedback_states=feedback_states,
            )
            selection = select_candidate(signals)
            eligible_selection = self._select_eligible_candidate(
                gate=FeedbackGate(session["feedback_gate"]),
                signals=signals,
                feedback_states=feedback_states,
            )
            self._repository.set_best_candidate(session_id, selection.selected_signal.candidate_id)
            metadata["selection_summary"] = selection.selection_summary
            metadata["selected_minor_robustness_risk_accepted"] = (
                selection.selected_signal.minor_robustness_risk_accepted
            )
            if selection.selected_signal.tradeoff_rationale is not None:
                metadata["selected_tradeoff_rationale"] = (
                    selection.selected_signal.tradeoff_rationale
                )
            else:
                metadata.pop("selected_tradeoff_rationale", None)
            metadata["eligible_best_candidate_id"] = (
                eligible_selection.selected_signal.candidate_id if eligible_selection else None
            )
            metadata["eligible_selection_summary"] = (
                eligible_selection.selection_summary if eligible_selection else None
            )
            metadata["eligible_minor_robustness_risk_accepted"] = (
                eligible_selection.selected_signal.minor_robustness_risk_accepted
                if eligible_selection
                else False
            )
        self._repository.update_session(session_id, metadata=metadata)
