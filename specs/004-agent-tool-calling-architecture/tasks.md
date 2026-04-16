# Tasks: Agent Tool-Calling Control Architecture

## Current Delivery Queue

- [x] T001 Build agentic experiment schema, loader, controller, broker, and service
- [x] T002 Add `rewardlab experiment` CLI surface (`validate/run/status/trace/stop/resume`)
- [x] T003 Add feedback submission and benchmark-run support
- [x] T004 Add baseline agentic unit coverage and verification report updates
- [x] T005 Pivot active runtime path to Gymnasium-only and remove Isaac active-path artifacts

## Stabilization And Gap Closure

- [x] T006 Fix green-gate regressions (`check_headers`, `mypy`, failing tests)
- [x] T007 Remove MCP execution path and consolidate local in-process tool execution
- [x] T008 Remove MCP policy/server spec fields and related tests/docs
- [x] T009 Wire control-mode resolver (`session_pipeline|agent_tools`) with compatibility behavior
- [x] T010 Enforce broker timeout/retry semantics from tool policy
- [x] T011 Implement runtime output controls from spec (`runtime_dir`, `report_detail`, `save_decision_trace`)
- [x] T012 Resolve allowlist/runtime mismatch with concrete tool implementations:
  - [x] `summarize_run_artifacts`
  - [x] `validate_reward_program`
- [x] T013 Add integration/e2e mixed-action Humanoid coverage for agentic loop
- [x] T014 Apply repo hygiene updates:
  - [x] `.gitignore` runtime artifact policy
  - [x] `src/rewardlab.egg-info` ignore policy
  - [x] optional `origin` remote URL modernization

## Final Validation

- [x] T015 Run full validation and record evidence in verification docs
