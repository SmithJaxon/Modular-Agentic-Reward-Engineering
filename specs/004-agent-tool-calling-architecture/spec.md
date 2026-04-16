# Feature Specification: Agent Tool-Calling Control Architecture

## Status

- Active feature branch target: `004-agent-tool-calling-architecture`
- Runtime scope: Gymnasium-only (`CartPole-v1` smoke, `Humanoid-v4` primary)
- Legacy compatibility: `rewardlab session ...` remains supported during migration

## Problem Statement

The existing `session` pipeline performs a fixed evaluate-reflect-revise loop.
This prevents the controller model from selecting and sequencing actions
dynamically based on evidence, budgets, and governance policy.

## Objectives

1. Enable autonomous controller decisions over explicit tool actions.
2. Keep all expensive work behind typed, auditable tool boundaries.
3. Enforce hard budgets and policy-gated stop behavior.
4. Preserve offline-safe and deterministic fallback behavior.
5. Keep Gymnasium real-execution path as the primary runtime.

## Functional Requirements

### FR-001 Controller Decision Loop

- The system SHALL maintain an autonomous loop that:
  - evaluates stop policy,
  - requests one next action from the controller,
  - executes the action via a broker,
  - persists decision and budget state.

### FR-002 Typed Tool Dispatch

- The system SHALL route controller actions through an explicit broker layer.
- The broker SHALL enforce tool allowlists and tool policy constraints.
- Tool execution SHALL return normalized `ToolResult` payloads.

### FR-003 Budget and Policy Enforcement

- Hard budget ceilings (tokens, USD, experiments, timesteps, wall-clock) SHALL
  stop runs when exhausted.
- Plateau/no-improve/failed-action thresholds SHALL be policy-enforced.
- Human-feedback actions SHALL honor governance gates and request budgets.

### FR-004 Experiment Spec Contract

- Users SHALL define autonomous runs through one spec file (`yaml` or `json`)
  covering objective, environment, models, budgets, governance, tool policy,
  execution settings, and outputs.

### FR-005 Decision Trace and Reporting

- The system SHALL persist experiment record, candidates, runs, decisions, and
  feedback artifacts.
- The system SHALL expose status and trace retrieval via CLI.

### FR-006 Local-Only Tool Execution

- The runtime SHALL execute agent actions through in-process local tool
  implementations when running on the experiment host machine.
- Action execution SHALL remain normalized through `ToolResult` contracts.
- No remote MCP server configuration SHALL be required for standard operation.

### FR-007 Control-Mode Compatibility

- The runtime SHALL expose an explicit control-mode selector:
  `session_pipeline` or `agent_tools`.
- Existing `session` behavior SHALL remain unchanged unless control mode
  explicitly opts into agentic behavior.

### FR-008 Tool-Policy Runtime Semantics

- `default_timeout_seconds` and `max_retries_per_tool` SHALL be enforced at
  broker execution time.
- `outputs.report_detail` and `outputs.save_decision_trace` SHALL influence
  report shape.
- `outputs.runtime_dir` SHALL be respected for report artifact location.

### FR-009 Tool Set Completeness

- The runtime SHALL provide executable support for all required allowlist tools:
  - `run_experiment`
  - `propose_reward_revision`
  - `estimate_cost_and_risk`
  - `compare_candidates`
  - `stop_or_continue_recommendation`
  - `summarize_run_artifacts`
  - `validate_reward_program`

## Non-Functional Requirements

- NFR-001: Offline-safe default behavior when model credentials are absent.
- NFR-002: Deterministic fallback path for tests and low-cost smoke workflows.
- NFR-003: Keep local artifacts/workflows worktree-scoped.
- NFR-004: Maintain lint/type/test quality gates.

## Acceptance Criteria

1. Autonomous runs can execute from a spec file and persist full trace evidence.
2. Controller can issue mixed action sequences with policy-valid outcomes.
3. Local tool execution is the default and only required action runtime path.
4. Tool-policy timeout/retry controls are enforced at runtime.
5. Validation suite passes (`check_headers`, `ruff`, `mypy`, offline pytest).
