# Agentic Tool-Calling Rework Plan

Status: in progress; Phases 0-6 implemented for current runtime on 2026-04-11.

## Objective

Replace the fixed iterative pipeline with a decision-turn orchestration model where:
- a primary optimizer agent decides what to do next,
- tool execution runs through worker agents with narrow task context,
- stopping is agent-driven using budget and plateau guidance,
- new tools can be added without changing orchestration flow logic.

## Design Principles

- No hardcoded experiment sequence in orchestration.
- Tool broker validates and routes requests, but does not dictate strategy.
- Primary agent owns strategy, tradeoffs, and stop decisions.
- Budget is multi-axis (time, compute, API tokens/cost, model usage).
- Everything important is traceable and replayable.

## Workstreams

### 1) Runtime Contracts

Add schemas for:
- `DecisionTurnState`
- `AgentDecision`
- `ToolRequest`
- `ToolResult`
- `BudgetState`
- `StopDecision`
- `RunTraceEvent`

Planned files:
- `src/rewardlab/schemas/agentic_run.py`
- `src/rewardlab/schemas/tool_contracts.py`
- `src/rewardlab/schemas/budget_state.py`

### 2) Primary Agent Runtime

Create long-lived primary agent controller with rich context memory.

Planned files:
- `src/rewardlab/agentic/primary_optimizer.py`
- `src/rewardlab/agentic/context_store.py`
- `src/rewardlab/agentic/decision_loop.py`

Responsibilities:
- read objective and current budget state
- generate next action plan
- issue zero or more tool calls
- decide continue/stop with rationale

### 3) Tool Broker + Registry (Modular)

Create a registry-driven broker so tools are pluggable.

Planned files:
- `src/rewardlab/tools/broker.py`
- `src/rewardlab/tools/registry.py`
- `src/rewardlab/tools/manifests/*.yaml`
- `src/rewardlab/tools/executors/*.py`

Responsibilities:
- validate request schema
- enforce budget/safety limits
- dispatch to executor
- normalize results for agent consumption

### 4) Worker Agent Execution

Add worker runner for isolated tool-call execution contexts.

Planned files:
- `src/rewardlab/agentic/worker_runner.py`
- `src/rewardlab/agentic/worker_context.py`

Responsibilities:
- build minimal prompt/context packet
- execute requested tool action
- return structured result and artifacts

### 5) Budget Engine (Multi-Axis)

Implement hard/soft budgets:
- wall-clock time
- training timesteps
- evaluation episodes
- API input/output tokens
- API USD spend
- per-model call quotas
- optional GPU-hours

Planned files:
- `src/rewardlab/agentic/budget_engine.py`
- `src/rewardlab/agentic/cost_model.py`

### 6) Stop Guidance (Not Hardcoded Flow)

Implement reusable stop heuristics that inform the primary agent:
- plateau detection
- diminishing return per cost
- objective threshold checks
- risk ceiling checks

Planned files:
- `src/rewardlab/agentic/stop_guidance.py`

Output rationale tags:
- `objective_met`
- `plateau`
- `cost_inefficient`
- `risk_limit`
- `manual`

### 7) Agent-First CLI

Add new commands:
- `rewardlab agent run --spec <file>`
- `rewardlab agent status --run-id <id>`
- `rewardlab agent events --run-id <id>`
- `rewardlab agent report --run-id <id>`

Planned files:
- `src/rewardlab/cli/agent_commands.py`
- `src/rewardlab/cli/app.py` (wire new command group)

### 8) Persistence + Reporting

Persist:
- decision-turn log
- tool-call log
- budget ledger snapshots
- best-candidate history
- stop rationale

Planned files:
- `src/rewardlab/persistence/agentic_repository.py`
- `src/rewardlab/orchestrator/reporting.py` (agentic sections)

### 9) Test Plan

Unit tests:
- broker validation and routing
- budget accounting
- stop guidance behavior
- context compression

Integration tests:
- end-to-end `agent run` with mocked tools
- gymnasium tool execution via worker
- stop decisions under budget pressure

Planned files:
- `tests/unit/test_agentic_*.py`
- `tests/integration/test_agentic_*.py`

## Execution Phases

1. Foundation: contracts, broker skeleton, registry, budget state.
2. Primary-agent loop: decision turns, tool call dispatch, trace persistence.
3. Tool migration: wrap existing gymnasium execution as a broker tool.
4. Stop/cost intelligence: plateau + cost-aware guidance.
5. CLI + report: `agent run/status/events/report`.
6. Validation: unit/integration + real Humanoid smoke.

## Progress Notes (2026-04-11)

- Phase 4 stop/cost intelligence is now implemented through:
  - `src/rewardlab/agentic/stop_guidance.py`
  - `src/rewardlab/agentic/cost_model.py`
  - `src/rewardlab/agentic/primary_optimizer.py` integration
- Policy tightening added:
  - compare cadence controls in `DecisionPolicySpec`
  - risk-aware and cost-aware experiment override shaping in `PrimaryOptimizer`
  - per-call estimated budget checks in `ToolBroker` + `BudgetEngine`
- Primary-agent LLM planning added:
  - `src/rewardlab/agentic/llm_planner.py`
  - `PrimaryOptimizer` now attempts planner decisions first when
    `agent.planner_provider=openai`, with heuristic fallback
  - planner prompt includes objective, budget, enabled tools, recent context,
    and stop-hint guidance
  - decision trace rows now include `decision_source` (`heuristic` or `llm_openai`)
- Planner hardening added:
  - strict mode (`agent.planner_fallback_enabled=false`) to disable heuristic fallback
  - planner call usage accounting (input/output tokens, model-call usage)
    applied to run budget ledger
  - planner argument normalization for tool contract compatibility
  - planner retry/repair attempts (`agent.planner_max_retries`)
  - explicit tool argument contracts embedded in planner prompt
  - planner failure feedback rows persisted as `planner.validation_failed` events
    and summarized in reports via `planner_feedback_summary`
- Validation added:
  - `tests/unit/test_agentic_stop_guidance.py`
  - `tests/unit/test_agentic_primary_optimizer.py`
  - `tests/unit/test_agentic_llm_planner.py`
  - `tests/unit/test_openai_client.py` usage extraction coverage
  - targeted suite: `39 passed`
- Real smoke confirmation:
  - heuristic profile: `agentrun-fec2eddea6ec` (`objective_met`)
  - OpenAI planner profile: `agentrun-6ddabb563daa` (`objective_met`,
    includes `decision_source=llm_openai`)
  - strict planner profile: `agentrun-2e157a200402` (`manual` stop via
    `planner_fallback_enabled=false` guardrail)
- Ongoing enhancements are tracked in `specs/agentic-improvement-backlog.md`.

## Completion Criteria

- Primary agent can run a full optimization without a hardcoded loop order.
- Tool broker can register a new tool without orchestrator code changes.
- Run can stop autonomously with a machine-readable rationale.
- Reports contain decision trace, tool trace, budget trace, and best reward summary.
