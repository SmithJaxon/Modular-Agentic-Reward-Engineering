---

description: "Task list for real Gymnasium and Isaac experiment readiness"
---

# Tasks: Real Experiment Readiness

**Input**: Design documents from `/specs/003-real-experiment-readiness/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`  
**Tests**: Tests are REQUIRED because this tranche changes the runtime path from
offline heuristics to real backend execution and must preserve the existing
offline-safe suite while adding backend smoke evidence.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependency)
- **[Story]**: Which user story the task belongs to (`US1`, `US2`, `US3`)
- Include exact file paths in descriptions

## Path Conventions

- Source: `src/rewardlab/`
- Tests: `tests/unit/`, `tests/contract/`, `tests/integration/`, `tests/e2e/`
- Tooling: `tools/quality/`, `tools/fixtures/`, `tools/reward_hack_probes/`
- Feature docs: `specs/003-real-experiment-readiness/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare reproducible backend-ready fixtures, validation markers, and
operator instructions without changing machine-global state.

- [x] T001 Update operator prerequisites and approval-gated install guidance in `README.md` and `specs/003-real-experiment-readiness/quickstart.md`
- [ ] T002 With user approval, install the required real-backend dependencies into `.venv\` and record the exact commands and detected versions in `specs/003-real-experiment-readiness/quickstart.md`
- [x] T003 [P] Add backend-specific pytest markers and skip guidance in `pyproject.toml` and `tests/conftest.py`
- [x] T004 [P] Add real-run fixture configs for Gymnasium and Isaac execution in `tools/fixtures/experiments/gymnasium_cartpole.json` and `tools/fixtures/experiments/isaac_default.json`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared execution layer, persistence, and artifact plumbing
needed by both backends.

**CRITICAL**: No real backend user story is complete until this phase is done.

- [x] T005 Implement reward program loading and validation in `src/rewardlab/experiments/reward_program.py`
- [x] T006 [P] Implement run artifact bundle writing in `src/rewardlab/experiments/artifacts.py`
- [x] T007 Implement actual experiment execution coordination in `src/rewardlab/experiments/execution_service.py`
- [x] T008 [P] Extend experiment run metadata for real execution mode and runtime status in `src/rewardlab/schemas/experiment_run.py` and `src/rewardlab/schemas/runtime_status.py`
- [x] T009 Implement persistence helpers for experiment runs and stored robustness assessments in `src/rewardlab/persistence/session_repository.py`
- [x] T010 [P] Add foundational unit coverage for reward loading, artifact manifests, and experiment-run persistence in `tests/unit/test_real_execution_foundations.py`

**Checkpoint**: Shared real-execution modules are available for backend integration.

---

## Phase 3: User Story 1 - Real Gymnasium Experiment Execution (Priority: P1)

**Goal**: Replace heuristic-only session stepping with a real Gymnasium-backed
experiment path that produces persisted evidence.

**Independent Test**: Start a Gymnasium session, run `session step`, and confirm
that a real environment run, persisted experiment run, and report evidence are
created through the normal lifecycle.

### Tests for User Story 1

- [x] T011 [P] [US1] Add contract tests for real Gymnasium environment resolution and actionable runtime failures in `tests/contract/test_gymnasium_backend_runtime.py`
- [x] T012 [P] [US1] Add integration coverage for session-step real Gymnasium experiment persistence in `tests/integration/test_gymnasium_real_experiment.py`
- [x] T013 [P] [US1] Add an end-to-end CLI smoke for actual Gymnasium execution in `tests/e2e/test_gymnasium_actual_experiment.py`

### Implementation for User Story 1

- [x] T014 [P] [US1] Implement real Gymnasium environment creation and runtime checks in `src/rewardlab/experiments/backends/gymnasium_backend.py`
- [x] T015 [P] [US1] Implement the minimal Gymnasium experiment runner in `src/rewardlab/experiments/gymnasium_runner.py`
- [x] T016 [US1] Integrate real Gymnasium execution into `src/rewardlab/orchestrator/session_service.py` and `src/rewardlab/orchestrator/iteration_engine.py`
- [x] T017 [US1] Persist Gymnasium run evidence and expose it in `src/rewardlab/orchestrator/reporting.py`
- [x] T018 [US1] Update the CLI workflow docs for actual Gymnasium sessions in `README.md` and `specs/001-iterative-reward-design/quickstart.md`

**Checkpoint**: Real Gymnasium experiments run through the normal session lifecycle.

---

## Phase 4: User Story 2 - Real Robustness Evidence and Review Artifacts (Priority: P2)

**Goal**: Route robustness and review evidence through the real backend run path.

**Independent Test**: Run a real session with robustness enabled and verify that
probe runs, stored assessments, and feedback artifacts all point to actual run
evidence.

### Tests for User Story 2

- [ ] T019 [P] [US2] Add integration coverage for lifecycle-triggered robustness execution and persisted assessments in `tests/integration/test_real_robustness_pipeline.py`
- [ ] T020 [P] [US2] Add integration coverage for actual rollout artifact bundles and feedback attachment in `tests/integration/test_real_demo_artifacts.py`

### Implementation for User Story 2

- [ ] T021 [US2] Integrate `RobustnessRunner` into the real session lifecycle in `src/rewardlab/orchestrator/session_service.py` and `src/rewardlab/experiments/robustness_runner.py`
- [ ] T022 [US2] Implement actual artifact capture and manifest emission in `src/rewardlab/experiments/artifacts.py` and `src/rewardlab/feedback/demo_artifacts.py`
- [ ] T023 [US2] Update robustness-aware selection and reporting to consume stored assessment evidence in `src/rewardlab/selection/policy.py`, `src/rewardlab/selection/risk_analyzer.py`, and `src/rewardlab/orchestrator/reporting.py`

**Checkpoint**: Final recommendation flow uses real robustness evidence and actual run artifacts.

---

## Phase 5: User Story 3 - Real Isaac Gym Experiment Execution (Priority: P3)

**Goal**: Execute the same actual experiment path on `isaacgym` with actionable
runtime readiness checks.

**Independent Test**: Start an Isaac session, run `session step`, and confirm
that a real Isaac experiment run and report are produced or that a precise
runtime prerequisite error is surfaced.

### Tests for User Story 3

- [ ] T024 [P] [US3] Add contract tests for Isaac runtime detection and prerequisite errors in `tests/contract/test_isaacgym_backend_runtime.py`
- [ ] T025 [P] [US3] Add integration coverage for actual Isaac experiment persistence in `tests/integration/test_isaac_real_experiment.py`
- [ ] T026 [P] [US3] Add an end-to-end CLI smoke for actual Isaac execution in `tests/e2e/test_isaac_actual_experiment.py`

### Implementation for User Story 3

- [ ] T027 [P] [US3] Implement real Isaac environment creation and readiness reporting in `src/rewardlab/experiments/backends/isaacgym_backend.py`
- [ ] T028 [P] [US3] Implement the Isaac experiment runner and execution-service integration in `src/rewardlab/experiments/isaacgym_runner.py` and `src/rewardlab/experiments/execution_service.py`
- [ ] T029 [US3] Integrate Isaac runtime configuration, reporting, and operator guidance in `src/rewardlab/orchestrator/session_service.py`, `src/rewardlab/orchestrator/reporting.py`, and `specs/003-real-experiment-readiness/quickstart.md`

**Checkpoint**: Real Isaac experiments run through the shared lifecycle or fail with precise prerequisite guidance.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, reproducibility, cleanup, and handoff refresh.

- [ ] T030 [P] Add backend smoke wrappers and documentation-friendly validation commands in `tools/quality/run_real_backend_smokes.ps1` and `tools/quality/run_full_validation.ps1`
- [x] T031 Run the approved real Gymnasium smoke validation and record the evidence in `specs/003-real-experiment-readiness/verification-report.md`
- [ ] T032 Run the approved real Isaac smoke validation and record the evidence in `specs/003-real-experiment-readiness/verification-report.md`
- [ ] T033 Perform dead-code cleanup and remove superseded MVP-only execution branches in `src/rewardlab/`, `tests/`, and `tools/`
- [ ] T034 Update `NEXT_AGENT_HANDOFF.md` and `specs/003-real-experiment-readiness/checklists/requirements.md` with final completion status and any remaining blocker notes

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies, but `T002` is approval-gated.
- Foundational (Phase 2): Depends on Phase 1 docs/fixture preparation and provides the shared real-execution base.
- US1 (Phase 3): Depends on Phase 2 and should be completed before US2 and US3 integration.
- US2 (Phase 4): Depends on US1 because it needs real primary run evidence.
- US3 (Phase 5): Depends on Phase 2 and may proceed in parallel with late US2 work if file ownership is disjoint, but should follow US1 architecture patterns.
- Polish (Phase 6): Depends on the stories needed for final readiness evidence.

### User Story Dependencies

- US1 (P1): First real execution milestone and MVP for this follow-on tranche.
- US2 (P2): Builds on real execution evidence from US1.
- US3 (P3): Uses the same execution layer once Gymnasium patterns are proven.

### Within Each User Story

- Tests first
- Backend/runtime support before orchestration integration
- Orchestration integration before reporting/docs
- Header/docstring and cleanup updates before story completion

## Parallel Opportunities

- Phase 1: `T003` and `T004` can run in parallel; `T002` waits for approval.
- Phase 2: `T006`, `T008`, and `T010` can run in parallel with care.
- US1: `T011`, `T012`, and `T013` can run in parallel; `T014` and `T015` can run in parallel.
- US2: `T019` and `T020` can run in parallel.
- US3: `T024`, `T025`, and `T026` can run in parallel; `T027` and `T028` can run in parallel if write sets stay disjoint.
- Polish: `T030` can proceed before final smoke validation tasks.

## Implementation Strategy

### Real Backend MVP First

1. Finish the shared execution layer.
2. Deliver actual Gymnasium execution and reporting.
3. Fold robustness and real artifacts into that path.
4. Extend the same path to Isaac Gym.

### Approval-Aware Delivery

1. Complete all code and tests that do not require new packages first.
2. Ask for approval only when the next critical path truly depends on install or runtime setup.
3. Record exact approved commands and versions once installs happen.

### Multi-Agent Delivery

1. Main agent owns shared execution integration and final verification.
2. Sub-agents can own isolated runners, backend-specific tests, or documentation updates.
3. Every sub-agent diff is reviewed like a pull request before merge.

## Notes

- This backlog supersedes additional feature work under `001-iterative-reward-design`; `001` is complete for the MVP/offline-safe tranche.
- Real readiness is not complete until both Gymnasium and Isaac have actual run evidence or only an explicit approval/runtime blocker remains.
- Keep all work within the dedicated worktree and follow the approval gates defined in `AGENTS.md` and `specs/002-autonomous-project-pass/`.
