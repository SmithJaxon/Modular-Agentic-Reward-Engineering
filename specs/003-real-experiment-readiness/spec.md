# Feature Specification: Real Experiment Readiness

**Feature Branch**: `003-real-experiment-readiness`  
**Created**: 2026-04-02  
**Status**: In Progress  
**Input**: User direction: "Remove Isaac for now and make the real target a full Gymnasium Humanoid pipeline with PPO-based final evaluation shaped like the EUREKA paper."

## User Scenarios & Testing

### User Story 1 - Real Gymnasium Smoke Execution (Priority: P1)

As a reinforcement learning researcher, I want `session step` to execute a real
Gymnasium-backed experiment so that candidate ranking is based on actual
environment interaction rather than text heuristics.

**Independent Test**: Start a `CartPole-v1` Gymnasium session in
`actual_backend` mode, run `session step`, and verify that the run persists
metrics, artifacts, and report evidence.

### User Story 2 - Real Robustness And Review Evidence (Priority: P2)

As a reinforcement learning researcher, I want robustness checks and review
artifacts to be derived from real Gymnasium runs so that candidate selection is
grounded in persisted evidence.

**Independent Test**: Run a real Gymnasium session with robustness enabled and
verify that probe runs, a stored `RobustnessAssessment`, and reviewable
artifacts are attached to the resulting candidate.

### User Story 3 - Gymnasium Humanoid PPO Evaluation (Priority: P3)

As a reinforcement learning researcher, I want Gymnasium Humanoid candidates to
be evaluated with PPO training so that final performance is measured in the same
shape as the EUREKA paper instead of by a single rollout heuristic.

**Independent Test**: With approved PPO dependencies installed in `.venv`,
start a `Humanoid-v4` session, run `session step`, and verify that the system:

1. Trains PPO for the configured timesteps budget.
2. Evaluates 10 checkpoints per training run.
3. Repeats training for 5 seeds.
4. Stores the average of the per-run best checkpoint mean `x_velocity` as the
   candidate score and report evidence.

## Edge Cases

- Gymnasium is installed but `Humanoid-v4` cannot be resolved.
- `Humanoid-v4` is available but `stable_baselines3` is not installed.
- Candidate reward code requires unsupported parameters for Gymnasium
  transitions.
- PPO training succeeds for some seeds but not all seeds.
- Rendering is unavailable; metrics-first evidence is still acceptable.
- Offline validation must remain deterministic and must not require PPO.

## Requirements

### Functional Requirements

- **FR-001**: The active real backend scope MUST be Gymnasium only.
- **FR-002**: The codebase MUST NOT route normal execution through Isaac-related
  runtime branches, tests, tooling, or contracts in this worktree.
- **FR-003**: The system MUST preserve a deterministic offline-safe path for
  unit, contract, and low-cost integration validation.
- **FR-004**: Each actual Gymnasium candidate iteration MUST persist an
  `ExperimentRun` with metrics and artifact references.
- **FR-005**: Real robustness execution MUST continue to run through the shared
  Gymnasium execution pipeline and persist a `RobustnessAssessment`.
- **FR-006**: Gymnasium Humanoid evaluation MUST use PPO training when the
  configured environment id is `Humanoid-v4` or `Humanoid-v5`.
- **FR-007**: The Gymnasium Humanoid PPO score MUST be computed as the average
  across 5 training runs of the best checkpoint mean `x_velocity` observed over
  10 evaluation checkpoints.
- **FR-008**: The PPO timesteps budget and checkpoint settings MUST be
  configurable via worktree-local environment variables.
- **FR-009**: If PPO prerequisites are missing, the system MUST fail with a
  clear prerequisite message and MUST NOT silently fall back to the rollout
  heuristic for Humanoid.
- **FR-010**: Operator docs MUST provide a reproducible Gymnasium Humanoid run
  path using checked-in fixtures and approval-gated `.venv` installs only.

### Constitution Alignment Requirements

- **CAR-001**: Gymnasium rollout logic and Humanoid PPO logic MUST stay inside
  focused execution modules rather than the session service.
- **CAR-002**: Docs, tasks, and handoff artifacts MUST describe Gymnasium-only
  scope and current PPO blockers accurately.
- **CAR-003**: Cleanup work MUST remove superseded Isaac-specific branches from
  active code, tests, and tooling.

## Key Entities

- **Actual Experiment Run**: A persisted real Gymnasium candidate evaluation.
- **Humanoid PPO Evaluation Protocol**: The PPO training and checkpoint scoring
  procedure used for `Humanoid-v4` and `Humanoid-v5`.
- **Run Artifact Bundle**: The manifest, metrics, traces, and optional media for
  a completed real run.
- **Backend Runtime Status**: The Gymnasium readiness or prerequisite summary
  reported for actual execution.

## Success Criteria

- **SC-001**: The documented Gymnasium smoke path completes a real `CartPole-v1`
  run from `session start` through `session report` with non-empty artifacts.
- **SC-002**: The documented Gymnasium Humanoid path produces PPO-based scoring
  metrics with 5-run, 10-checkpoint aggregation once approved PPO dependencies
  are installed.
- **SC-003**: No active code, tests, tooling, or schemas in this worktree
  require Isaac to run.
- **SC-004**: Reports produced by actual Gymnasium runs clearly distinguish
  single-rollout evidence from Humanoid PPO evidence.

## Assumptions

- `gymnasium`, `mujoco`, and `torch` may already be present locally, but any
  additional PPO dependency installation remains approval-gated.
- `x_velocity` from Gymnasium Humanoid step info is an acceptable task fitness
  metric for the Mujoco Humanoid path because the EUREKA appendix reports
  forward velocity for that environment.
