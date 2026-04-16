# Research: LLM-Guided Reward Function Iteration

## Decision 1: Use PyTorch for all ML model and tensor operations
- Decision: Standardize on PyTorch for model-facing and tensor-level workflows.
- Rationale: User preference is explicit, ecosystem maturity is strong for RL
  experimentation, and PyTorch integrates well with existing RL tooling.
- Alternatives considered: JAX (high performance but added migration complexity),
  TensorFlow (less aligned with project constraints).

## Decision 2: Keep the system as a local-first modular CLI pipeline
- Decision: Use a CLI-driven orchestrator with explicit session lifecycle commands.
- Rationale: Supports reproducible experiments, scripting, and contract testing
  without introducing web-service complexity in v1.
- Alternatives considered: Web dashboard first (higher delivery overhead), notebook-only
  workflow (weaker automation and weaker contract verification).

## Decision 3: Separate orchestration context from execution/review contexts
- Decision: Run experiments and peer review in isolated worker contexts and share
  only bounded summaries back to the orchestrator.
- Rationale: Matches spec requirement for context efficiency and reduces risk of
  long-running experiment artifacts polluting orchestration state.
- Alternatives considered: Single-context loop (simpler but violates isolation goal).

## Decision 4: Implement iterative evaluation as three experiment classes
- Decision: Support `performance`, `reflection`, and `robustness` experiment classes.
- Rationale: Matches project goals and EUREKA-style iterative improvement with
  explicit anti-reward-hacking checks.
- Alternatives considered: Performance-only loop (insufficient for robustness);
  monolithic experiment type (poor traceability).

## Decision 4b: Support dual environment backends via adapter interface
- Decision: Add backend adapters for `gymnasium`, selected by a
  required session parameter.
- Rationale: Enables direct comparability to EUREKA-style experiments while
  retaining flexibility to run lighter Gymnasium workflows when desired.
- Alternatives considered: Gymnasium-only (insufficient for target comparison);
  Gymnasium-only execution keeps the active runtime simpler and locally reproducible.

## Decision 5: Treat convergence and gating controls as explicit input parameters
- Decision: Require no-improvement threshold, max-iteration limit, and feedback
  gating policy at session start.
- Rationale: Avoids hidden magic constants and enables reproducible validation.
- Alternatives considered: Hardcoded defaults (faster setup but weaker auditability).

## Decision 6: Use robustness-aware selection guidance with agent discretion
- Decision: Selection combines primary performance, robustness, and feedback
  signals while allowing justified tradeoffs for minor robustness issues.
- Rationale: Aligns with user direction that final decisions should be agent-driven
  and rationale-backed, not rigidly thresholded.
- Alternatives considered: Hard robustness gate (too rigid), single-metric
  optimization (reward-hacking risk).

## Decision 7: Persist artifacts in append-only records plus resumable snapshots
- Decision: Store iteration events/results in append-only JSONL and session
  checkpoints in JSON plus SQLite metadata index.
- Rationale: Supports resumability, post-hoc analysis, and deterministic replay.
- Alternatives considered: In-memory only (fragile), database-only storage
  (unnecessary complexity for v1).

## Decision 8: Build verification tooling early, not as post-hoc work
- Decision: Include dedicated tools for deterministic fixtures, reward-hack probe
  scenarios, contract validation, and schema linting.
- Rationale: High-confidence iteration quality depends on frequent, cheap feedback.
- Alternatives considered: Manual verification only (high regression risk).

## Decision 9: Standardize quality gates around static + runtime checks
- Decision: Require ruff, mypy, pytest unit/integration/contract/e2e, and
  coverage checks in routine development.
- Rationale: Improves reliability and catches interface drift before long runs.
- Alternatives considered: Runtime tests only (late detection of defects).

## Decision 10: Handle model/API failures with bounded retries and pause/resume
- Decision: Apply bounded retry with backoff; if exhausted, pause session with
  resumable state and preserve best-known candidate evidence.
- Rationale: Prevents silent failure and aligns with clarified resilience policy.
- Alternatives considered: Fail-fast termination (poor long-run resilience),
  infinite retry (unbounded execution risk).

## Open Questions Resolved for Planning

All previously material ambiguities are resolved in the specification clarifications.
No remaining `NEEDS CLARIFICATION` items are required before implementation tasks.
