"""
Summary: Primary optimizer policy for agentic decision-turn execution.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import re

from rewardlab.agentic.context_store import ContextStore
from rewardlab.agentic.llm_planner import (
    DecisionPlanner,
    LLMDecisionPlanner,
    PlannerAttemptFeedback,
    PlannerCallUsage,
)
from rewardlab.agentic.stop_guidance import StopGuidance
from rewardlab.schemas.agentic_run import (
    AgentDecision,
    AgentDecisionAction,
    AgenticRunSpec,
    StopDecisionTag,
)
from rewardlab.schemas.budget_state import BudgetState
from rewardlab.schemas.tool_contracts import ToolResultStatus


class PrimaryOptimizer:
    """
    Make high-level strategy decisions for each decision turn.
    """

    def __init__(
        self,
        *,
        stop_guidance: StopGuidance | None = None,
        decision_planner: DecisionPlanner | None = None,
    ) -> None:
        """
        Initialize optimizer dependencies.
        """
        self._stop_guidance = stop_guidance or StopGuidance()
        self._decision_planner = decision_planner or LLMDecisionPlanner()
        self._latest_planner_feedback: tuple[PlannerAttemptFeedback, ...] = ()

    def decide(
        self,
        *,
        run_id: str,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> AgentDecision:
        """
        Generate one decision-turn action from current context and budgets.
        """
        self._latest_planner_feedback = ()
        remaining_calls = budget_state.remaining_calls_per_model()
        primary_calls_remaining = remaining_calls.get(spec.agent.primary_model)
        if primary_calls_remaining is not None and primary_calls_remaining <= 0:
            return AgentDecision(
                turn_index=turn_index,
                action=AgentDecisionAction.STOP,
                summary="Stopping because primary-model call quota is exhausted.",
                stop_tag=StopDecisionTag.COST_INEFFICIENT,
                stop_reason=(
                    f"primary model call quota exhausted for {spec.agent.primary_model}"
                ),
            )

        if not spec.tools.enabled:
            return AgentDecision(
                turn_index=turn_index,
                action=AgentDecisionAction.STOP,
                summary="Stopping because no tools are enabled for this run.",
                stop_tag=StopDecisionTag.MANUAL,
                stop_reason="tool allowlist is empty",
            )

        latest = context.latest_tool_result
        if latest is not None and latest.status in {
            ToolResultStatus.FAILED,
            ToolResultStatus.REJECTED,
        }:
            return AgentDecision(
                turn_index=turn_index,
                action=AgentDecisionAction.STOP,
                summary="Stopping after a non-recoverable tool failure.",
                stop_tag=StopDecisionTag.MANUAL,
                stop_reason=(
                    f"tool {latest.tool_name} returned {latest.status.value}"
                ),
            )

        planner_provider = spec.agent.planner_provider.strip().lower()
        if planner_provider == "openai":
            if self._planner_budget_exhausted(budget_state):
                return AgentDecision(
                    turn_index=turn_index,
                    action=AgentDecisionAction.STOP,
                    summary="Stopping because planner API budget is exhausted.",
                    stop_tag=StopDecisionTag.COST_INEFFICIENT,
                    stop_reason="planner API budget exhausted",
                )

        stop_decision = self._stop_guidance.evaluate(
            spec=spec,
            context=context,
            budget_state=budget_state,
        )
        planned_decision = self._decision_planner.plan(
            turn_index=turn_index,
            spec=spec,
            context=context,
            budget_state=budget_state,
            stop_hint=stop_decision,
        )
        self._latest_planner_feedback = self._planner_last_feedback()
        planner_usage = self._planner_last_usage()
        if planner_usage is not None:
            self._apply_planner_usage(
                budget_state=budget_state,
                usage=planner_usage,
                default_model=spec.agent.primary_model,
            )
        if planned_decision is not None:
            normalized_planned = self._normalize_planner_decision(
                decision=planned_decision,
                run_id=run_id,
                turn_index=turn_index,
                spec=spec,
                context=context,
                budget_state=budget_state,
            )
            if normalized_planned is not None:
                return self._maybe_export_before_stop(
                    decision=normalized_planned,
                    run_id=run_id,
                    spec=spec,
                    context=context,
                )
            self._latest_planner_feedback = (
                *self._latest_planner_feedback,
                PlannerAttemptFeedback(
                    attempt_index=0,
                    max_attempts=0,
                    failure_type="invalid_tool_arguments",
                    reason=(
                        "planner decision could not be normalized to local "
                        "tool contract"
                    ),
                    output_excerpt=None,
                ),
            )
            return self._maybe_export_before_stop(
                decision=planned_decision,
                run_id=run_id,
                spec=spec,
                context=context,
            )

        if planner_provider == "openai" and not spec.agent.planner_fallback_enabled:
            self._latest_planner_feedback = (
                *self._latest_planner_feedback,
                PlannerAttemptFeedback(
                    attempt_index=0,
                    max_attempts=0,
                    failure_type="planner_unavailable",
                    reason="strict mode blocked heuristic fallback after planner failure",
                    output_excerpt=None,
                ),
            )
            return self._maybe_export_before_stop(
                decision=AgentDecision(
                    turn_index=turn_index,
                    decision_source="planner_guard",
                    action=AgentDecisionAction.STOP,
                    summary="Stopping because strict planner mode disallows heuristic fallback.",
                    stop_tag=StopDecisionTag.MANUAL,
                    stop_reason="planner unavailable and planner_fallback_enabled is false",
                ),
                run_id=run_id,
                spec=spec,
                context=context,
            )

        if stop_decision is not None:
            return self._maybe_export_before_stop(
                decision=AgentDecision(
                    turn_index=turn_index,
                    action=AgentDecisionAction.STOP,
                    summary=stop_decision.summary,
                    stop_tag=stop_decision.tag,
                    stop_reason=stop_decision.reason,
                ),
                run_id=run_id,
                spec=spec,
                context=context,
            )

        latest_experiment = context.latest_experiment_result()
        if latest_experiment is not None and "run_probe_suite" in spec.tools.enabled:
            candidate_id_raw = latest_experiment.output.get("candidate_id")
            score_raw = latest_experiment.output.get("score")
            if (
                isinstance(candidate_id_raw, str)
                and candidate_id_raw
                and isinstance(score_raw, int | float)
                and candidate_id_raw not in context.probed_candidate_ids()
            ):
                probe_seed = latest_experiment.output.get("seed", spec.environment.seed)
                probe_index = latest_experiment.output.get("iteration_index", 0)
                probe_seed_value = (
                    int(probe_seed)
                    if isinstance(probe_seed, int | float)
                    else spec.environment.seed
                )
                probe_index_value = (
                    int(probe_index)
                    if isinstance(probe_index, int | float)
                    else 0
                )
                overrides = self._build_training_overrides(
                    spec=spec,
                    experiment_index=len(context.experiment_scores()),
                )
                return AgentDecision(
                    turn_index=turn_index,
                    action=AgentDecisionAction.REQUEST_TOOL,
                    summary=f"Requesting robustness probes for {candidate_id_raw}.",
                    tool_name="run_probe_suite",
                    tool_arguments={
                        "session_id": run_id,
                        "candidate_id": candidate_id_raw,
                        "primary_score": float(score_raw),
                        "environment_id": spec.environment.id,
                        "environment_backend": spec.environment.backend.value,
                        "objective_file": spec.objective.text_file,
                        "reward_file": spec.objective.baseline_reward_file,
                        "seed": probe_seed_value,
                        "iteration_index": probe_index_value,
                        "variant_label": "default",
                        "overrides": overrides,
                    },
                    tool_rationale=(
                        "Evaluate robustness risk before the next reward-search action."
                    ),
                )

        if "compare_candidates" in spec.tools.enabled:
            snapshots = context.candidate_snapshots()
            if self._should_compare(spec=spec, context=context, snapshots=snapshots):
                return AgentDecision(
                    turn_index=turn_index,
                    action=AgentDecisionAction.REQUEST_TOOL,
                    summary="Comparing current candidate set by aggregate score.",
                    tool_name="compare_candidates",
                    tool_arguments={"candidates": snapshots},
                    tool_rationale="Rank candidates periodically as new evidence accumulates.",
                )

        if "run_experiment" in spec.tools.enabled:
            return self._build_experiment_decision(
                run_id=run_id,
                turn_index=turn_index,
                spec=spec,
                context=context,
                budget_state=budget_state,
            )

        tool_name = spec.tools.enabled[0]
        return AgentDecision(
            turn_index=turn_index,
            action=AgentDecisionAction.REQUEST_TOOL,
            summary=f"Requesting fallback tool call: {tool_name}.",
            tool_name=tool_name,
            tool_arguments={},
            tool_rationale="Fallback to first enabled tool because run_experiment is unavailable.",
        )

    def _planner_last_usage(self) -> PlannerCallUsage | None:
        """
        Return usage from the current planner when the planner exposes it.
        """
        usage_getter = getattr(self._decision_planner, "last_usage", None)
        if not callable(usage_getter):
            return None
        usage = usage_getter()
        return usage if isinstance(usage, PlannerCallUsage) else None

    def _planner_last_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return planner failure feedback rows when planner exposes them.
        """
        feedback_getter = getattr(self._decision_planner, "last_feedback", None)
        if not callable(feedback_getter):
            return ()
        feedback_raw = feedback_getter()
        if not isinstance(feedback_raw, tuple | list):
            return ()
        rows: list[PlannerAttemptFeedback] = []
        for row in feedback_raw:
            if isinstance(row, PlannerAttemptFeedback):
                rows.append(row)
        return tuple(rows)

    def drain_planner_feedback(self) -> tuple[PlannerAttemptFeedback, ...]:
        """
        Return and clear planner feedback rows for the most recent decision turn.
        """
        rows = self._latest_planner_feedback
        self._latest_planner_feedback = ()
        return rows

    @staticmethod
    def _apply_planner_usage(
        *,
        budget_state: BudgetState,
        usage: PlannerCallUsage,
        default_model: str,
    ) -> None:
        """
        Apply planner-call token/cost/model usage to budget ledger state.
        """
        budget_state.usage.api_input_tokens += max(0, usage.api_input_tokens)
        budget_state.usage.api_output_tokens += max(0, usage.api_output_tokens)
        budget_state.usage.api_cost_usd += max(0.0, usage.api_cost_usd)
        model_used = usage.model_used or default_model
        if model_used:
            budget_state.usage.calls_per_model[model_used] = (
                budget_state.usage.calls_per_model.get(model_used, 0)
                + max(1, usage.call_count)
            )

    @staticmethod
    def _planner_budget_exhausted(budget_state: BudgetState) -> bool:
        """
        Report whether planner-related API budgets are exhausted.
        """
        if budget_state.max_api_usd > 0.0 and budget_state.remaining_api_usd() <= 0.0:
            return True
        if budget_state.max_api_input_tokens > 0 and budget_state.remaining_api_input_tokens() <= 0:
            return True
        if (
            budget_state.max_api_output_tokens > 0
            and budget_state.remaining_api_output_tokens() <= 0
        ):
            return True
        return False

    def _build_experiment_decision(
        self,
        *,
        run_id: str,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> AgentDecision:
        """
        Build a standard run-experiment decision payload.
        """
        experiment_index = len(context.experiment_scores())
        overrides = self._build_training_overrides(spec=spec, experiment_index=experiment_index)
        variant_label = "default"
        include_reflection = True

        risk_level = self._latest_probe_risk_level(context)
        if risk_level in {"medium", "high"}:
            variant_label = "risk_aware"
            learning_rate_raw = overrides.get("ppo_learning_rate")
            if isinstance(learning_rate_raw, int | float):
                overrides["ppo_learning_rate"] = float(learning_rate_raw) * 0.8
            evaluation_raw = overrides.get("evaluation_episodes")
            if isinstance(evaluation_raw, int | float):
                overrides["evaluation_episodes"] = int(evaluation_raw) + 1

        api_budget_fraction_used = self._api_budget_fraction_used(budget_state)
        if api_budget_fraction_used >= 0.7:
            variant_label = "cost_aware" if variant_label == "default" else variant_label
            evaluation_raw = overrides.get("evaluation_episodes")
            if isinstance(evaluation_raw, int | float):
                overrides["evaluation_episodes"] = max(1, int(evaluation_raw) - 1)
            reflection_raw = overrides.get("reflection_episodes")
            if isinstance(reflection_raw, int | float):
                overrides["reflection_episodes"] = max(0, int(reflection_raw) - 1)
        if api_budget_fraction_used >= 0.85:
            include_reflection = False

        remaining_eval = budget_state.remaining_evaluation_episodes()
        eval_requested = overrides.get("evaluation_episodes")
        if isinstance(eval_requested, int | float):
            overrides["evaluation_episodes"] = max(
                1,
                min(int(eval_requested), max(1, remaining_eval)),
            )
        if remaining_eval <= 1:
            include_reflection = False

        candidate_id = f"cand-{run_id[-6:]}-{experiment_index:03d}"
        return AgentDecision(
            turn_index=turn_index,
            action=AgentDecisionAction.REQUEST_TOOL,
            summary="Requesting exploratory tool call: run_experiment.",
            tool_name="run_experiment",
            tool_arguments={
                "session_id": run_id,
                "candidate_id": candidate_id,
                "environment_id": spec.environment.id,
                "environment_backend": spec.environment.backend.value,
                "objective_file": spec.objective.text_file,
                "reward_file": spec.objective.baseline_reward_file,
                "seed": spec.environment.seed + experiment_index,
                "iteration_index": experiment_index,
                "variant_label": variant_label,
                "include_reflection": include_reflection,
                "overrides": overrides,
            },
            tool_rationale=(
                "Run an environment experiment via the broker and use score trend"
                " signals for the next decision."
            ),
        )

    def _normalize_planner_decision(
        self,
        *,
        decision: AgentDecision,
        run_id: str,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> AgentDecision | None:
        """
        Normalize planner-produced tool arguments to valid local tool contracts.
        """
        if decision.action is not AgentDecisionAction.REQUEST_TOOL:
            return decision
        tool_name = decision.tool_name
        if tool_name is None:
            return None
        if tool_name == "run_experiment":
            return self._normalize_run_experiment_decision(
                decision=decision,
                run_id=run_id,
                turn_index=turn_index,
                spec=spec,
                context=context,
                budget_state=budget_state,
            )
        if tool_name == "run_probe_suite":
            return self._normalize_run_probe_suite_decision(
                decision=decision,
                run_id=run_id,
                spec=spec,
                context=context,
            )
        if tool_name == "compare_candidates":
            args = dict(decision.tool_arguments)
            snapshots = context.candidate_snapshots()
            if not snapshots:
                return None
            args["candidates"] = snapshots
            return decision.model_copy(update={"tool_arguments": args})
        if tool_name == "export_report":
            args = dict(decision.tool_arguments)
            if "output_path" not in args:
                args["output_path"] = f".rewardlab/agentic_exports/{run_id}.final.json"
            if "report_payload" not in args:
                args["report_payload"] = self._build_export_payload(run_id=run_id, context=context)
            return decision.model_copy(update={"tool_arguments": args})
        return decision

    def _normalize_run_experiment_decision(
        self,
        *,
        decision: AgentDecision,
        run_id: str,
        turn_index: int,
        spec: AgenticRunSpec,
        context: ContextStore,
        budget_state: BudgetState,
    ) -> AgentDecision:
        """
        Convert planner run_experiment arguments into full contract-compliant payload.
        """
        base_decision = self._build_experiment_decision(
            run_id=run_id,
            turn_index=turn_index,
            spec=spec,
            context=context,
            budget_state=budget_state,
        )
        args = dict(base_decision.tool_arguments)
        planner_args = dict(decision.tool_arguments)
        aliases = self._normalize_environment_aliases(planner_args)
        planner_args.update(aliases)

        candidate_id_raw = planner_args.pop("candidate_name", None)
        if "candidate_id" not in planner_args and isinstance(candidate_id_raw, str):
            normalized_name = candidate_id_raw.strip().replace(" ", "-")
            if normalized_name:
                planner_args["candidate_id"] = normalized_name

        training_timesteps = planner_args.pop("training_timesteps", None)
        evaluation_episodes = planner_args.pop("evaluation_episodes", None)
        include_reflection = planner_args.pop("include_reflection", None)
        planner_overrides = planner_args.get("overrides")
        merged_overrides = dict(args.get("overrides", {}))
        if isinstance(planner_overrides, dict):
            for key, value in planner_overrides.items():
                merged_overrides[key] = value
        if isinstance(training_timesteps, int | float):
            merged_overrides["ppo_total_timesteps"] = max(64, int(training_timesteps))
        if isinstance(evaluation_episodes, int | float):
            merged_overrides["evaluation_episodes"] = max(1, int(evaluation_episodes))
        args["overrides"] = merged_overrides
        if isinstance(include_reflection, bool):
            args["include_reflection"] = include_reflection

        allowed_overrides = {
            "candidate_id",
            "session_id",
            "seed",
            "iteration_index",
            "variant_label",
            "environment_id",
            "environment_backend",
        }
        for key in allowed_overrides:
            value = planner_args.get(key)
            if value is None:
                continue
            if key in {
                "candidate_id",
                "session_id",
                "variant_label",
                "environment_id",
                "environment_backend",
            }:
                if isinstance(value, str) and value.strip():
                    args[key] = value.strip()
                continue
            if key in {"seed", "iteration_index"} and isinstance(value, int | float):
                args[key] = int(value)

        # Keep objective/reward file paths pinned to spec defaults to prevent invalid overrides.
        args["objective_file"] = spec.objective.text_file
        args["reward_file"] = spec.objective.baseline_reward_file

        return decision.model_copy(update={"tool_arguments": args})

    def _normalize_run_probe_suite_decision(
        self,
        *,
        decision: AgentDecision,
        run_id: str,
        spec: AgenticRunSpec,
        context: ContextStore,
    ) -> AgentDecision | None:
        """
        Convert planner run_probe_suite arguments into full contract-compliant payload.
        """
        args = dict(decision.tool_arguments)
        latest = context.latest_experiment_result()
        if latest is not None:
            latest_candidate = latest.output.get("candidate_id")
            latest_score = latest.output.get("score")
            latest_seed = latest.output.get("seed")
            latest_iteration = latest.output.get("iteration_index")
            if "candidate_id" not in args and isinstance(latest_candidate, str):
                args["candidate_id"] = latest_candidate
            if "primary_score" not in args and isinstance(latest_score, int | float):
                args["primary_score"] = float(latest_score)
            if "seed" not in args and isinstance(latest_seed, int | float):
                args["seed"] = int(latest_seed)
            if "iteration_index" not in args and isinstance(latest_iteration, int | float):
                args["iteration_index"] = int(latest_iteration)
        if "candidate_id" not in args or "primary_score" not in args:
            return None
        aliases = self._normalize_environment_aliases(args)
        args.update(aliases)
        args.setdefault("session_id", run_id)
        args.setdefault("environment_id", spec.environment.id)
        args.setdefault("environment_backend", spec.environment.backend.value)
        args["objective_file"] = spec.objective.text_file
        args["reward_file"] = spec.objective.baseline_reward_file
        args.setdefault("variant_label", "default")
        args.setdefault("seed", spec.environment.seed)
        args.setdefault("iteration_index", len(context.experiment_scores()) - 1)
        if "overrides" not in args:
            args["overrides"] = self._build_training_overrides(
                spec=spec,
                experiment_index=len(context.experiment_scores()),
            )
        if not isinstance(args.get("overrides"), dict):
            return None
        return decision.model_copy(update={"tool_arguments": args})

    @staticmethod
    def _normalize_environment_aliases(arguments: dict[str, object]) -> dict[str, object]:
        """
        Extract environment backend/id aliases from planner-friendly fields.
        """
        normalized: dict[str, object] = {}
        env_compound = arguments.get("environment")
        if isinstance(env_compound, str):
            text = env_compound.strip()
            if text:
                match = re.match(r"^([a-zA-Z0-9_-]+)\s*/\s*([a-zA-Z0-9_.:-]+)$", text)
                if match:
                    normalized["environment_backend"] = match.group(1)
                    normalized["environment_id"] = match.group(2)
        if "env_id" in arguments and "environment_id" not in normalized:
            env_id = arguments.get("env_id")
            if isinstance(env_id, str) and env_id.strip():
                normalized["environment_id"] = env_id.strip()
        if "env_backend" in arguments and "environment_backend" not in normalized:
            env_backend = arguments.get("env_backend")
            if isinstance(env_backend, str) and env_backend.strip():
                normalized["environment_backend"] = env_backend.strip()
        env_id_raw = arguments.get("environment_id")
        if isinstance(env_id_raw, str):
            text = env_id_raw.strip()
            match = re.match(r"^([a-zA-Z0-9_-]+)\s*/\s*([a-zA-Z0-9_.:-]+)$", text)
            if match:
                normalized["environment_backend"] = match.group(1)
                normalized["environment_id"] = match.group(2)
        return normalized

    def _maybe_export_before_stop(
        self,
        *,
        decision: AgentDecision,
        run_id: str,
        spec: AgenticRunSpec,
        context: ContextStore,
    ) -> AgentDecision:
        """
        Convert stop decisions into final-export tool calls when export is enabled.
        """
        if decision.action is not AgentDecisionAction.STOP:
            return decision
        if "export_report" not in spec.tools.enabled:
            return decision
        if context.export_completed_count() > 0:
            return decision
        return AgentDecision(
            turn_index=decision.turn_index,
            decision_source=decision.decision_source,
            action=AgentDecisionAction.REQUEST_TOOL,
            summary="Exporting final decision summary before stopping.",
            tool_name="export_report",
            tool_arguments={
                "output_path": f".rewardlab/agentic_exports/{run_id}.final.json",
                "report_payload": self._build_export_payload(run_id=run_id, context=context),
            },
            tool_rationale="Persist a standalone final report artifact before termination.",
        )

    @staticmethod
    def _api_budget_fraction_used(budget_state: BudgetState) -> float:
        """
        Return consumed API USD fraction in [0, 1+] for budget-aware exploration.
        """
        if budget_state.max_api_usd <= 0.0:
            return 0.0
        return max(0.0, budget_state.usage.api_cost_usd / budget_state.max_api_usd)

    @staticmethod
    def _latest_probe_risk_level(context: ContextStore) -> str | None:
        """
        Return the latest robustness probe risk level when available.
        """
        latest_probe = context.latest_probe_result()
        if latest_probe is None:
            return None
        risk_level_raw = latest_probe.output.get("risk_level")
        if not isinstance(risk_level_raw, str):
            return None
        normalized = risk_level_raw.strip().lower()
        return normalized if normalized else None

    @staticmethod
    def _should_compare(
        *,
        spec: AgenticRunSpec,
        context: ContextStore,
        snapshots: list[dict[str, object]],
    ) -> bool:
        """
        Decide whether to dispatch a compare-candidates tool call this turn.
        """
        min_candidates = spec.decision.min_candidates_before_compare
        if len(snapshots) < min_candidates:
            return False
        if spec.decision.require_probe_before_compare:
            latest_candidate_raw = snapshots[-1].get("candidate_id")
            if not isinstance(latest_candidate_raw, str) or not latest_candidate_raw:
                return False
            if latest_candidate_raw not in context.probed_candidate_ids():
                return False
        completed = context.compare_completed_count()
        compare_interval = spec.decision.compare_every_new_candidates
        compare_threshold = min_candidates + (completed * compare_interval)
        return len(snapshots) >= compare_threshold

    @staticmethod
    def _build_training_overrides(
        *,
        spec: AgenticRunSpec,
        experiment_index: int,
    ) -> dict[str, object]:
        """
        Build training override dictionary shared by experiment/probe tools.
        """
        return {
            "execution_mode": spec.training_defaults.execution_mode,
            "llm_provider": spec.training_defaults.llm_provider,
            "llm_model": spec.training_defaults.llm_model,
            "ppo_total_timesteps": spec.training_defaults.ppo_total_timesteps,
            "ppo_num_envs": spec.training_defaults.ppo_num_envs,
            "ppo_n_steps": spec.training_defaults.ppo_n_steps,
            "ppo_batch_size": spec.training_defaults.ppo_batch_size,
            "ppo_learning_rate": spec.training_defaults.ppo_learning_rate,
            "ppo_gamma": spec.training_defaults.ppo_gamma,
            "ppo_gae_lambda": spec.training_defaults.ppo_gae_lambda,
            "ppo_clip_range": spec.training_defaults.ppo_clip_range,
            "ppo_n_epochs": spec.training_defaults.ppo_n_epochs,
            "ppo_ent_coef": spec.training_defaults.ppo_ent_coef,
            "ppo_vf_coef": spec.training_defaults.ppo_vf_coef,
            "ppo_max_grad_norm": spec.training_defaults.ppo_max_grad_norm,
            "ppo_activation_fn": spec.training_defaults.ppo_activation_fn,
            "ppo_policy_hidden_sizes": list(spec.training_defaults.ppo_policy_hidden_sizes),
            "evaluation_episodes": spec.training_defaults.evaluation_episodes,
            "reflection_episodes": spec.training_defaults.reflection_episodes,
            "reflection_interval_steps": spec.training_defaults.reflection_interval_steps,
            "train_seed": spec.environment.seed + experiment_index,
        }

    @staticmethod
    def _build_export_payload(
        *,
        run_id: str,
        context: ContextStore,
    ) -> dict[str, object]:
        """
        Build a lightweight exported summary payload from context state.
        """
        return {
            "run_id": run_id,
            "best_score": context.best_score,
            "candidate_snapshots": context.candidate_snapshots(),
            "decision_count": len(context.decisions),
            "tool_result_count": len(context.tool_results),
        }
