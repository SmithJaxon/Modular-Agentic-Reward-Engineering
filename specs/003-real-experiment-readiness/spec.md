# Feature Specification: Real Experiment Readiness

**Feature Branch**: `003-real-experiment-readiness`  
**Created**: 2026-04-02  
**Status**: Planned  
**Input**: User description: "Rework our planning to include all of these requirements so that a new agent can keep iterating until I can run an actual experiment in both gymnasium and isaac gym."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Real Gymnasium Experiment Execution (Priority: P1)

As a reinforcement learning researcher, I want `session step` to execute a real
Gymnasium-backed experiment for each reward candidate so that candidate ranking
is based on actual environment interaction rather than text heuristics.

**Why this priority**: Gymnasium is the lightest real backend in the project
scope and is the fastest path from the current MVP to a true experiment loop.

**Independent Test**: With approved local dependencies installed in the
worktree `.venv`, start a Gymnasium session, run `session step`, and verify that
the system instantiates a real environment, produces a persisted experiment run,
stores metrics and artifacts, and includes that evidence in the exported report.

**Acceptance Scenarios**:

1. **Given** a session configured for `gymnasium`, **When** `session step` runs,
   **Then** the system executes at least one real backend-driven experiment for
   the current candidate and persists the resulting metrics and artifact
   references.
2. **Given** a candidate reward definition is invalid or a backend experiment
   fails, **When** retries are exhausted, **Then** the session pauses with
   resumable state and the failure is recorded with an actionable reason.

---

### User Story 2 - Real Robustness Evidence and Review Artifacts (Priority: P2)

As a reinforcement learning researcher, I want robustness checks and human
review artifacts to be generated from real backend runs so that final candidate
selection reflects real behavioral evidence instead of placeholder summaries.

**Why this priority**: Without real robustness evidence and actual review
artifacts, the recommendation path still falls short of the intended research
workflow even if primary execution is real.

**Independent Test**: Run a session with real Gymnasium execution and verify
that robustness probes execute through the backend pipeline, persist a
robustness assessment, and generate reviewable rollout artifacts that can be
attached to feedback.

**Acceptance Scenarios**:

1. **Given** a completed primary experiment run, **When** robustness checks are
   enabled, **Then** the system executes probe variants through the real backend
   path and records a robustness assessment tied to the candidate.
2. **Given** a completed candidate run, **When** feedback is requested or
   submitted, **Then** the system references a real artifact bundle from the run
   rather than a placeholder-only review file.

---

### User Story 3 - Real Isaac Gym Experiment Execution (Priority: P3)

As a reinforcement learning researcher, I want the same actual experiment path
to run on `isaacgym` so that the project can execute real experiments on both
supported backends.

**Why this priority**: The full product promise is not met until both
backends execute real experiments through the shared session lifecycle.

**Independent Test**: With approved local runtime dependencies available in the
worktree `.venv`, start an `isaacgym` session, run `session step`, and verify
that the system executes a real Isaac-backed experiment, persists metrics and
artifacts, and exports a report with that evidence.

**Acceptance Scenarios**:

1. **Given** a session configured for `isaacgym` and a supported environment,
   **When** `session step` runs, **Then** the system executes a real backend
   experiment and records evidence through the same reporting pipeline used for
   Gymnasium.
2. **Given** the Isaac runtime is unavailable or misconfigured, **When** the
   user attempts a real Isaac experiment, **Then** the system surfaces an
   actionable prerequisite error instead of silently falling back to fake or
   heuristic execution.

### Edge Cases

- Worktree-local backend dependencies have not yet been approved for download.
- A real environment backend is installed but the requested `environment_id` is
  unsupported or fails to initialize.
- Candidate reward code cannot be loaded or raises during experiment execution.
- A run can execute metrics-only evidence but cannot render frames in the current
  environment.
- Robustness probe variants partially fail while the primary run succeeds.
- Isaac-specific runtime requirements differ from Gymnasium and need explicit
  local configuration after approval.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The normal experiment path for real sessions MUST execute actual
  backend-driven environment interaction and MUST NOT derive candidate ranking
  solely from text heuristics or injected fake environments.
- **FR-002**: The system MUST preserve a deterministic offline-safe path for
  unit and contract validation, but that path MUST be clearly separated from the
  actual experiment path and MUST NOT be mistaken for a real run in reports.
- **FR-003**: Each candidate iteration in a real session MUST produce at least
  one persisted `ExperimentRun` record containing backend, environment, status,
  timing, metrics, and artifact references.
- **FR-004**: Candidate reward definitions MUST be loadable through an explicit
  execution interface so they can be evaluated against real backend transitions.
- **FR-005**: `session step` MUST route real experiment execution through the
  selected backend adapter and use the resulting evidence to update candidate
  ranking and report content.
- **FR-006**: The Gymnasium backend MUST support at least one checked-in,
  documented real experiment target that can be exercised from the worktree with
  approved local dependencies.
- **FR-007**: Robustness execution MUST run through the real backend experiment
  pipeline and persist a `RobustnessAssessment` associated with the candidate.
- **FR-008**: Human feedback bundles MUST include references to actual run
  artifacts produced by backend execution, such as metrics files, frame sets,
  videos, or equivalent run evidence.
- **FR-009**: Session reports MUST identify whether evidence came from actual
  backend execution, include run identifiers and artifact references, and
  surface robustness outcomes alongside the final recommendation.
- **FR-010**: Backend execution failures MUST use bounded retry and backoff; if
  retries are exhausted, the session MUST pause with resumable state and clear
  failure evidence.
- **FR-011**: The Isaac Gym backend MUST support actual environment creation and
  experiment execution when the required runtime is available in the approved
  worktree-local environment.
- **FR-012**: When Isaac runtime prerequisites are missing, the system MUST
  surface an actionable error that identifies the missing prerequisite and MUST
  NOT silently downgrade to a fake execution path.
- **FR-013**: The validation plan MUST include offline regression coverage plus
  backend-specific smoke validation for real Gymnasium execution and real Isaac
  execution once approved dependencies are available.
- **FR-014**: Operator documentation MUST provide reproducible commands for
  running at least one actual Gymnasium experiment and one actual Isaac Gym
  experiment from the worktree.
- **FR-015**: Any dependency installation, backend runtime setup, or external
  asset acquisition required for this feature MUST remain subject to the
  existing approval gate and worktree-local installation restrictions.

### Constitution Alignment Requirements

- **CAR-001**: Real experiment execution MUST be introduced through focused
  modules for reward loading, backend execution, artifact capture, and
  orchestration integration rather than by collapsing logic into the session
  service.
- **CAR-002**: New planning artifacts MUST identify the concrete modules,
  schemas, persistence paths, and tests needed for real backend execution.
- **CAR-003**: All new or modified Python files and routines touched by this
  tranche MUST satisfy the file-header and method-docstring constitution rules.
- **CAR-004**: The final readiness tranche MUST include explicit cleanup of the
  MVP-only execution shortcuts or stale branches that are superseded by the real
  experiment path.

### Key Entities *(include if feature involves data)*

- **Actual Experiment Run**: A persisted record of a candidate evaluation that
  came from a real backend environment rather than a fake or heuristic source.
- **Reward Program**: The executable reward definition derived from a candidate
  and loaded into the experiment runtime through a constrained interface.
- **Run Artifact Bundle**: The metrics, manifests, frame outputs, logs, and
  other files produced by a real backend run and stored under the runtime data
  directory.
- **Backend Runtime Status**: A structured summary of whether a backend is ready,
  misconfigured, or blocked by a missing approved dependency.
- **Stored Robustness Assessment**: A persisted risk summary derived from probe
  runs executed through the actual backend pipeline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A documented Gymnasium smoke command completes a real backend run
  from `session start` through `session report` and produces non-empty run
  metrics and artifact references in 100% of validated attempts.
- **SC-002**: A documented Isaac Gym smoke command completes a real backend run
  from `session start` through `session report` and produces non-empty run
  metrics and artifact references in 100% of validated attempts once approved
  local runtime prerequisites are present.
- **SC-003**: Session reports produced by real runs clearly distinguish actual
  backend evidence from offline test-mode evidence in 100% of validated cases.
- **SC-004**: Robustness and feedback artifacts are derived from actual run
  evidence for the selected candidate in at least one validated end-to-end path.
- **SC-005**: All dependency setup needed for the feature is reproducible inside
  the worktree `.venv` with 0 machine-level installs, PATH edits, or writes
  outside the workspace.

## Assumptions

- The current `001-iterative-reward-design` implementation provides the correct
  session, reporting, feedback, and persistence skeleton but still requires a
  real execution layer.
- A minimal real experiment is acceptable for initial readiness as long as it
  uses a real backend environment, produces persisted run evidence, and can be
  repeated from documented commands.
- The user will approve any required backend dependency downloads before the
  corresponding installation step runs.
- If a backend cannot render media in the current environment, a metrics-first
  artifact bundle with other run evidence is still acceptable for initial
  readiness, provided the run itself is real.
