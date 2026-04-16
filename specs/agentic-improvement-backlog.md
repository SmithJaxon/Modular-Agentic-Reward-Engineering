# Agentic Improvement Backlog

Last updated: 2026-04-11

This file is the living list of improvements for the agentic tool-calling runtime.
Update this file whenever a new issue, idea, or hardening task is discovered.

## Implemented

- [x] Decision-trace provenance (`decision_source`) for `heuristic` vs `llm_openai`.
- [x] OpenAI planner mode with context-rich prompt and heuristic fallback.
- [x] Strict planner mode (`planner_fallback_enabled: false`) for fail-closed behavior.
- [x] Planner argument normalization for local tool contracts.
- [x] Planner usage accounting into budget ledger:
  - model-call usage
  - input/output token usage
  - API cost usage when provided by SDK response
- [x] Planner retry/repair loop (`planner_max_retries`) for malformed planner output.
- [x] Per-tool argument contracts included in planner prompt.
- [x] Per-call compute budget estimate checks in tool broker.
- [x] Planner-attempt feedback traces:
  - per-attempt failure type (`parse_error`, `schema_validation_error`, etc.)
  - persisted `planner.validation_failed` events in run event stream
  - report-level summary counts in `planner_feedback_summary`
- [x] Extended PPO configuration support in agentic training defaults:
  - policy architecture (`ppo_policy_hidden_sizes`, `ppo_activation_fn`)
  - optimization knobs (`ppo_gamma`, `ppo_gae_lambda`, `ppo_clip_range`,
    `ppo_n_epochs`, `ppo_ent_coef`, `ppo_vf_coef`, `ppo_max_grad_norm`)
- [x] Gymnasium robotics runtime registration hardening:
  - `gymnasium_runtime` now auto-imports optional `gymnasium_robotics` when installed
- [x] Planner argument normalization hardening:
  - compound environment aliases accepted in `environment_id` (e.g. `gymnasium/Env-v1`)
  - `run_probe_suite` file paths pinned to spec defaults (`objective_file`, `reward_file`)
  - `compare_candidates` payload normalized to context candidate snapshots

## High Priority Next

- [ ] Add one-turn planner retry cap by failure type:
  - parse/schema failures retry
  - disallowed-tool failures retry
  - transport/API failures configurable retry/backoff
- [ ] Add PPO-mode OpenAI planner validation profile and regression tests.
- [ ] Add integration test asserting minimum `llm_openai` decision ratio in OpenAI mode.
- [ ] Add fail-closed policy option for invalid planner tool arguments:
  - `normalize` (current behavior)
  - `reject_and_retry`
  - `reject_and_stop`

## Medium Priority

- [ ] Parallel tool execution honoring `max_parallel_workers` with safe conflict controls.
- [ ] Worker isolation upgrade from in-process runner to stronger process boundary.
- [ ] Budget-aware planner prompt augmentation:
  - estimated cost impact per candidate action
  - explicit "most expensive remaining dimension" indicator
- [ ] Adaptive compare cadence tuned by uncertainty/risk trend.
- [ ] Structured report section summarizing planner retries and repair outcomes.

## Lower Priority

- [ ] Add richer prompt templates per environment family (humanoid, ant, cartpole).
- [ ] Add optional memory compression strategy for long decision traces.
- [ ] Add optional tool cooldown controls to avoid repeated low-value tool calls.

## Notes

- Existing legacy session-mode roadmap tasks remain tracked in
  `specs/001-iterative-reward-design/tasks.md` (including `T064` Isaac Gym validation).
- This backlog focuses on the new agentic architecture path.
