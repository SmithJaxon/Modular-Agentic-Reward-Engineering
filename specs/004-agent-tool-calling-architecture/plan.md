# Implementation Plan: Agent Tool-Calling Control Architecture

**Branch**: `004-agent-tool-calling-architecture`  
**Date**: 2026-04-10  
**Status**: Planned

## Summary

Replace the current fixed `session step` iterative pipeline controller with an
agent-driven control loop where:

1. A primary controller agent decides what to do next.
2. All expensive actions execute through explicit tools.
3. Each tool call runs in an isolated worker context (not the controller context).
4. Stop/continue is decided by the controller with explicit budget and plateau policy guidance.

The immediate target remains Gymnasium, with `Humanoid-v4` as the main real
experiment path and `CartPole-v1` as the low-cost smoke path.

## Why This Rework

The current orchestration flow is deterministic and pipeline-driven:

- `plan iteration -> execute run -> build reflection -> create candidate`

This does not let the model choose when/how to run experiments, when to spend
budget, or when to stop. The new architecture moves those decisions into the
controller agent while keeping strong runtime guardrails.

## Goals

- Agent chooses next action based on objective, evidence, budget state, and trend signals.
- Tool invocation is explicit, typed, auditable, and isolated from controller context.
- Runtime supports multi-dimensional budgets:
  - time
  - API tokens/cost
  - compute/execution count/timesteps
  - optional human-feedback request budget
- Stopping behavior is autonomous but policy-constrained.
- Backward-compatible fallback to existing mode during rollout.

## Non-Goals (Phase 1)

- No multi-backend expansion beyond Gymnasium.
- No full UI work; CLI-first surface remains the primary operator interface.
- No removal of existing `session` commands until new path is validated.

## User-Facing Experiment Definition

Experiments are defined in a single spec file (`yaml` preferred). This is the
control contract the agent follows.

Core sections:

- `objective`
- `environment`
- `baseline_reward`
- `models`
- `budgets`
- `governance` (plateau/stopping policy + feedback policy)
- `tool_policy`
- `execution` (backend-specific tunables like PPO)
- `outputs`

Reference examples:

- `tools/fixtures/experiments/agent_humanoid_balanced.yaml`
- `tools/fixtures/experiments/agent_humanoid_high_budget.yaml`
- `tools/fixtures/experiments/agent_cartpole_lowcost.yaml`

## Proposed CLI Surface

New commands (additive):

- `rewardlab experiment validate --file <spec>`
- `rewardlab experiment run --file <spec> --json`
- `rewardlab experiment status --experiment-id <id> --json`
- `rewardlab experiment trace --experiment-id <id> --json`
- `rewardlab experiment stop --experiment-id <id> --json`
- `rewardlab experiment resume --experiment-id <id> --json`

Existing `session` commands remain available during migration.

## Architecture

### 1) Controller Loop (Primary Agent)

New module:

- `src/rewardlab/agentic/controller_loop.py`

Responsibilities:

- Build controller context summary from persisted state.
- Ask controller model for next action.
- Validate action against policy and allowed tools.
- Dispatch through tool broker.
- Incorporate tool outputs into state.
- Re-evaluate stop criteria.

Controller outputs a strict action object:

- `action`: `run_experiment | propose_reward | analyze | compare | request_feedback | stop`
- `rationale`
- `expected_value`
- `expected_cost`
- `inputs`

### 2) Tool Broker (Isolated Execution Boundary)

New module:

- `src/rewardlab/agentic/tool_broker.py`

Responsibilities:

- Route controller actions to tool executors.
- Enforce per-tool timeout/retry policy.
- Spawn worker execution with minimal context payload.
- Normalize tool responses into structured result envelopes.

### 3) Tool Executors (Worker Context)

New package:

- `src/rewardlab/agentic/tools/`

Initial tool set:

- `run_experiment` (wraps existing real backend execution path)
- `summarize_run_artifacts`
- `propose_reward_revision` (model-backed reward design tool)
- `validate_reward_program`
- `compare_candidates`
- `estimate_cost_and_risk`
- `stop_or_continue_recommendation`
- `request_human_feedback` (policy-gated)

### 4) State and Trace

Persist new experiment records:

- experiment config snapshot
- action decisions and rationales
- tool invocations + outputs
- budget ledger snapshots per step
- stop decision evidence

Proposed modules:

- `src/rewardlab/schemas/agent_experiment.py`
- `src/rewardlab/persistence/decision_trace_store.py`

### 5) Policy and Budget Engine

New module:

- `src/rewardlab/agentic/policy_engine.py`

Responsibilities:

- Hard-limit enforcement for all configured budgets.
- Soft-threshold warnings surfaced to controller context.
- Plateau and diminishing-return signal computation.
- Gate human-feedback tools when disallowed or exhausted.

### 6) Compatibility and Rollout

Feature flag:

- `REWARDLAB_CONTROL_MODE=session_pipeline|agent_tools`

Default remains `session_pipeline` until stability gates pass.

## Controller Context Strategy

Controller context stays rich but compact. It should include:

- objective and environment summary
- baseline and top-N candidate summaries
- recent run trend table (scores, variance, failures)
- budget state and projected spend risk
- current plateau/diminishing-return signals
- last few decision outcomes

It should avoid large raw traces by default. Raw artifacts remain available to
workers on demand.

## Stopping Policy (Default Guidance)

Controller should stop when one or more of:

- hard budget exhausted
- plateau over `plateau_window` with `min_relative_improvement` unmet
- max no-improve streak reached
- repeated failed actions exceed threshold
- projected expected gain is lower than configured cost threshold

Policy engine can force stop even if controller asks to continue.

## Migration Plan

### Phase 1: Spec + Contracts + Scaffolding

- Add experiment spec schema + parser + validation CLI.
- Add tool request/response contracts.
- Add controller loop skeleton with mocked tool execution.

### Phase 2: Toolization of Existing Execution

- Wrap current Gymnasium execution path as `run_experiment` tool.
- Wrap reward program validator as tool.
- Add artifact summarizer tool.

### Phase 3: Decision Loop + Policy Enforcement

- Integrate real controller model calls.
- Add budget ledger and policy engine.
- Add autonomous stop/continue behavior.

### Phase 4: Human Feedback and Advanced Analysis

- Add policy-gated human feedback tool.
- Add candidate comparison and run-risk analysis tools.

### Phase 5: Hardening

- End-to-end Humanoid tests in `agent_tools` mode.
- Resumability + interruption handling.
- Report output with full decision trace.

## Testing Strategy

- Unit:
  - spec parsing/validation
  - tool contract validation
  - policy/budget decision logic
- Integration:
  - controller loop with mocked tool outputs
  - tool broker retry/timeout behavior
- End-to-end:
  - CartPole low-cost autonomous run
  - Humanoid PPO autonomous run with autonomous stop
- Regression:
  - existing `session_pipeline` behavior remains intact

## Acceptance Criteria

- User can define and run experiment exclusively through a spec file.
- Controller autonomously selects different tool actions across the run.
- At least one real Humanoid run completes in `agent_tools` mode with full trace.
- Controller stops autonomously for policy-valid reason without manual intervention.
- Reports include:
  - best candidate
  - score trends
  - budget usage
  - decision trace with rationale

## Risks and Mitigations

- Risk: controller thrashes between actions.
  - Mitigation: policy guardrails + expected value scoring + cooldown logic.
- Risk: context window bloat.
  - Mitigation: compact summaries + worker-context isolation.
- Risk: cost drift.
  - Mitigation: hard ceilings + per-action spend projection + forced stop.
- Risk: migration instability.
  - Mitigation: feature-flag rollout, keep existing path intact until parity.

