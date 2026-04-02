---

description: "Task list template for feature implementation"
---

# Tasks: LLM-Guided Reward Function Iteration

**Input**: Design documents from `/specs/001-iterative-reward-design/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED for this feature because the specification and user direction emphasize strong verification, modular testable chunks, and contract confidence.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.
When code is changed, tasks MUST include constitution-compliance work for file
headers, function/method headers, and dead-code cleanup.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Source: `src/rewardlab/`
- Tests: `tests/unit/`, `tests/contract/`, `tests/integration/`, `tests/e2e/`
- Tooling: `tools/quality/`, `tools/fixtures/`, `tools/reward_hack_probes/`
- Feature docs: `specs/001-iterative-reward-design/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project skeleton, quality tooling, and repeatable local verification.

- [x] T001 Create package scaffolding and module init files in `src/rewardlab/__init__.py`, `src/rewardlab/cli/__init__.py`, `src/rewardlab/orchestrator/__init__.py`, `src/rewardlab/llm/__init__.py`, `src/rewardlab/experiments/__init__.py`, `src/rewardlab/experiments/backends/__init__.py`, `src/rewardlab/feedback/__init__.py`, `src/rewardlab/selection/__init__.py`, `src/rewardlab/persistence/__init__.py`, `src/rewardlab/schemas/__init__.py`, and `src/rewardlab/utils/__init__.py`
- [x] T002 Initialize dependency and project metadata in `pyproject.toml`
- [x] T003 Configure linting, formatting, typing, and pytest tooling in `pyproject.toml`
- [x] T004 [P] Configure test discovery and markers in `pytest.ini`
- [x] T005 [P] Add runtime environment variable template in `.env.example`
- [x] T006 [P] Implement contract/schema validation utility in `tools/quality/validate_contracts.py`
- [x] T007 [P] Implement file-header and docstring audit utility in `tools/quality/check_headers.py`
- [x] T008 [P] Create deterministic objective and reward fixture seeds in `tools/fixtures/objectives/cartpole.txt` and `tools/fixtures/rewards/cartpole_baseline.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core session abstractions and shared infrastructure that every story depends on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T009 Implement session config validation model in `src/rewardlab/schemas/session_config.py`
- [ ] T010 [P] Implement session report validation model in `src/rewardlab/schemas/session_report.py`
- [ ] T011 Implement SQLite metadata store in `src/rewardlab/persistence/sqlite_store.py`
- [ ] T012 [P] Implement append-only JSONL event logger in `src/rewardlab/persistence/event_log.py`
- [ ] T013 Implement session repository facade in `src/rewardlab/persistence/session_repository.py`
- [ ] T014 [P] Implement bounded retry and backoff utilities in `src/rewardlab/utils/retry.py`
- [ ] T015 Implement orchestrator state machine transitions in `src/rewardlab/orchestrator/state_machine.py`
- [ ] T016 [P] Implement backend adapter interface in `src/rewardlab/experiments/backends/base.py`
- [ ] T017 [P] Implement OpenAI API client wrapper with env-var credentials in `src/rewardlab/llm/openai_client.py`
- [ ] T018 Implement foundational unit tests for schemas, retry logic, and state transitions in `tests/unit/test_foundational_components.py`

**Checkpoint**: Foundation ready; user stories can now be implemented.

---

## Phase 3: User Story 1 - Iterative Reward Optimization Loop (Priority: P1) MVP

**Goal**: Deliver end-to-end iterative loop execution with stop/interruption behavior and best-candidate export.

**Independent Test**: Start a session, run multiple iterations, interrupt safely, and verify best-known candidate export with evidence.

### Tests for User Story 1

- [ ] T019 [P] [US1] Add CLI contract tests for `session start`, `session step`, and `session stop` in `tests/contract/test_session_lifecycle_cli.py`
- [ ] T020 [P] [US1] Add integration test for iterative evaluate-reflect-revise ranking loop in `tests/integration/test_iteration_loop.py`
- [ ] T021 [P] [US1] Add end-to-end interruption and best-candidate export test in `tests/e2e/test_interrupt_best_candidate.py`

### Implementation for User Story 1

- [ ] T022 [P] [US1] Implement reward candidate schema and mapping in `src/rewardlab/schemas/reward_candidate.py`
- [ ] T023 [P] [US1] Implement reflection record schema and mapping in `src/rewardlab/schemas/reflection_record.py`
- [ ] T024 [US1] Implement session lifecycle service in `src/rewardlab/orchestrator/session_service.py`
- [ ] T025 [US1] Implement iteration engine (evaluate -> reflect -> revise) in `src/rewardlab/orchestrator/iteration_engine.py`
- [ ] T026 [US1] Implement baseline multi-signal selection policy in `src/rewardlab/selection/policy.py`
- [ ] T027 [US1] Implement session CLI command handlers in `src/rewardlab/cli/session_commands.py`
- [ ] T028 [US1] Implement CLI app bootstrap and JSON response wiring in `src/rewardlab/cli/app.py`
- [ ] T029 [US1] Implement checkpoint and resume flow in `src/rewardlab/orchestrator/checkpointing.py`
- [ ] T030 [US1] Implement best-candidate report generation in `src/rewardlab/orchestrator/reporting.py`
- [ ] T031 [US1] Wire per-iteration event emission and persisted evidence hooks in `src/rewardlab/persistence/event_log.py`
- [ ] T032 [US1] Add required file headers and function/method docstrings in `src/rewardlab/orchestrator/session_service.py` and `src/rewardlab/orchestrator/iteration_engine.py`

**Checkpoint**: User Story 1 is independently testable and can serve as MVP baseline.

---

## Phase 4: User Story 2 - Reward Hacking and Overfitting Detection (Priority: P2)

**Goal**: Add cross-backend robustness experiments and explicit reward-hacking risk analysis.

**Independent Test**: Run robustness probes on both `gymnasium` and `isaacgym` backends and verify risk flags and selection rationale.

### Tests for User Story 2

- [ ] T033 [P] [US2] Add backend adapter contract tests for Gymnasium and Isaac Gym in `tests/contract/test_backend_adapters.py`
- [ ] T034 [P] [US2] Add integration tests for reward-hacking probe matrix in `tests/integration/test_reward_hack_probes.py`
- [ ] T035 [P] [US2] Add integration test for `environment_backend` routing in `tests/integration/test_backend_selection.py`

### Implementation for User Story 2

- [ ] T036 [P] [US2] Implement Gymnasium backend adapter in `src/rewardlab/experiments/backends/gymnasium_backend.py`
- [ ] T037 [P] [US2] Implement Isaac Gym backend adapter in `src/rewardlab/experiments/backends/isaacgym_backend.py`
- [ ] T038 [US2] Implement backend adapter factory and resolver in `src/rewardlab/experiments/backends/factory.py`
- [ ] T039 [P] [US2] Implement experiment run schema in `src/rewardlab/schemas/experiment_run.py`
- [ ] T040 [US2] Implement robustness experiment runner with architecture/hyperparameter variants in `src/rewardlab/experiments/robustness_runner.py`
- [ ] T041 [US2] Implement robustness assessment schema and risk summary logic in `src/rewardlab/schemas/robustness_assessment.py`
- [ ] T042 [US2] Implement risk analyzer and tradeoff rationale capture in `src/rewardlab/selection/risk_analyzer.py`
- [ ] T043 [US2] Create reward-hacking probe matrix configuration in `tools/reward_hack_probes/probe_matrix.yaml`
- [ ] T044 [US2] Integrate robustness outputs into final candidate policy in `src/rewardlab/selection/policy.py`
- [ ] T045 [US2] Add required file headers and function/method docstrings in `src/rewardlab/experiments/robustness_runner.py` and `src/rewardlab/selection/risk_analyzer.py`

**Checkpoint**: Robustness and reward-hacking detection work across both backends.

---

## Phase 5: User Story 3 - Human and External Peer Feedback (Priority: P3)

**Goal**: Integrate human and peer feedback channels with configurable gating rules for final recommendations.

**Independent Test**: Submit human feedback, request peer feedback from isolated context, and verify gating behavior influences final recommendation.

### Tests for User Story 3

- [ ] T046 [P] [US3] Add CLI contract tests for `feedback submit-human` and `feedback request-peer` in `tests/contract/test_feedback_cli.py`
- [ ] T047 [P] [US3] Add integration tests for session-level feedback gating modes in `tests/integration/test_feedback_gating.py`
- [ ] T048 [P] [US3] Add integration tests for conflicting human/peer feedback resolution in `tests/integration/test_feedback_conflicts.py`

### Implementation for User Story 3

- [ ] T049 [P] [US3] Implement feedback entry schema in `src/rewardlab/schemas/feedback_entry.py`
- [ ] T050 [US3] Implement human feedback ingestion service in `src/rewardlab/feedback/human_feedback_service.py`
- [ ] T051 [US3] Implement isolated peer feedback client in `src/rewardlab/feedback/peer_feedback_client.py`
- [ ] T052 [US3] Implement feedback gating evaluator in `src/rewardlab/feedback/gating.py`
- [ ] T053 [US3] Implement visual demonstration artifact tracker in `src/rewardlab/feedback/demo_artifacts.py`
- [ ] T054 [US3] Implement feedback CLI command handlers in `src/rewardlab/cli/feedback_commands.py`
- [ ] T055 [US3] Integrate feedback signals into iteration summaries in `src/rewardlab/orchestrator/iteration_engine.py`
- [ ] T056 [US3] Integrate gating and feedback requirements into final recommendation path in `src/rewardlab/orchestrator/session_service.py`
- [ ] T057 [US3] Add required file headers and function/method docstrings in `src/rewardlab/feedback/human_feedback_service.py` and `src/rewardlab/feedback/peer_feedback_client.py`

**Checkpoint**: Feedback channels and gating policies are independently testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification hardening, cleanup, and evidence capture.

- [ ] T058 [P] Update operator documentation and examples in `README.md` and `specs/001-iterative-reward-design/quickstart.md`
- [ ] T059 Implement one-command validation runner in `tools/quality/run_full_validation.ps1`
- [ ] T060 [P] Implement end-to-end cycle-time budget test in `tests/e2e/test_iteration_cycle_budget.py`
- [ ] T061 [P] Implement schema contract regression tests in `tests/unit/test_schema_contracts.py`
- [ ] T062 [P] Implement header/docstring audit tests in `tests/unit/test_header_audit.py`
- [ ] T063 Run mandatory dead-code cleanup pass for `src/rewardlab/` and `tests/`
- [ ] T064 Execute full quality gate and record evidence in `specs/001-iterative-reward-design/verification-report.md`
- [ ] T065 Update completion notes and compliance status in `specs/001-iterative-reward-design/checklists/requirements.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies.
- Foundational (Phase 2): Depends on Setup completion; blocks all user stories.
- User Stories (Phases 3-5): Depend on Foundational completion.
- Polish (Phase 6): Depends on completion of selected user stories.

### User Story Dependencies

- US1 (P1): Can start immediately after Phase 2 and defines MVP behavior.
- US2 (P2): Depends on US1 iteration loop primitives and selection baseline.
- US3 (P3): Depends on US1 session lifecycle and report generation paths.

### Within Each User Story

- Tests first for the story phase.
- Schema/model changes before service logic.
- Service logic before CLI wiring.
- Update file headers and function/method headers before story completion.

## Parallel Opportunities

- Setup: T004, T005, T006, T007, T008 can run in parallel after T003.
- Foundational: T010, T012, T014, T016, T017 can run in parallel after T009.
- US1: T019, T020, T021 can run in parallel; T022 and T023 can run in parallel.
- US2: T033, T034, T035 can run in parallel; T036 and T037 can run in parallel.
- US3: T046, T047, T048 can run in parallel; T049 can run in parallel with T050/T051.
- Polish: T058, T060, T061, T062 can run in parallel before T064.

---

## Parallel Example: User Story 1

```bash
# Run US1 test tasks in parallel:
Task: "T019 [US1] tests/contract/test_session_lifecycle_cli.py"
Task: "T020 [US1] tests/integration/test_iteration_loop.py"
Task: "T021 [US1] tests/e2e/test_interrupt_best_candidate.py"

# Build US1 schema components in parallel:
Task: "T022 [US1] src/rewardlab/schemas/reward_candidate.py"
Task: "T023 [US1] src/rewardlab/schemas/reflection_record.py"
```

## Parallel Example: User Story 2

```bash
# Run US2 contract/integration tests in parallel:
Task: "T033 [US2] tests/contract/test_backend_adapters.py"
Task: "T034 [US2] tests/integration/test_reward_hack_probes.py"
Task: "T035 [US2] tests/integration/test_backend_selection.py"

# Implement backend adapters in parallel:
Task: "T036 [US2] src/rewardlab/experiments/backends/gymnasium_backend.py"
Task: "T037 [US2] src/rewardlab/experiments/backends/isaacgym_backend.py"
```

## Parallel Example: User Story 3

```bash
# Run US3 tests in parallel:
Task: "T046 [US3] tests/contract/test_feedback_cli.py"
Task: "T047 [US3] tests/integration/test_feedback_gating.py"
Task: "T048 [US3] tests/integration/test_feedback_conflicts.py"

# Implement feedback channels in parallel:
Task: "T050 [US3] src/rewardlab/feedback/human_feedback_service.py"
Task: "T051 [US3] src/rewardlab/feedback/peer_feedback_client.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup).
2. Complete Phase 2 (Foundational).
3. Complete Phase 3 (US1).
4. Validate interruption-safe best-candidate export and iteration loop.

### Incremental Delivery

1. Ship US1 for baseline orchestration.
2. Add US2 for robustness and reward-hacking detection across backends.
3. Add US3 for human/peer feedback and gating.
4. Complete Phase 6 polish and evidence capture.

### Two-Agent Comparison Strategy

1. Use this single `tasks.md` as the shared scope baseline.
2. Step-by-step agent executes tasks in strict ID order.
3. One-shot agent may batch tasks, but must report completion by original task IDs for comparability.

---

## Notes

- [P] tasks indicate no direct file conflict and no unfinished prerequisite dependency.
- Story labels map each task directly to one user story for independent verification.
- Each user story phase includes tests and implementation work so it can be validated in isolation.
- This task plan intentionally front-loads verification tooling to maximize implementation feedback quality.
