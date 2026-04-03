# Phase 5 Handoff: Human and External Peer Feedback

**Status**: Completed on 2026-04-02 and superseded by `specs/001-iterative-reward-design/phase6-handoff.md`  
**Prepared**: 2026-04-02  
**Feature**: `001-iterative-reward-design`  
**Prepared from implementation branch**: `001-iterative-reward-design-impl-p4`  
**Target integration branch for next work**: `iterative-main`

## 1. Scope for the Next Thread

Implement Phase 5 / US3 from `tasks.md`:

- `T046` feedback CLI contract tests
- `T047` feedback gating integration tests
- `T048` conflicting feedback integration tests
- `T049` feedback entry schema
- `T050` human feedback ingestion service
- `T051` isolated peer feedback client
- `T052` feedback gating evaluator
- `T053` visual demonstration artifact tracker
- `T054` feedback CLI command handlers
- `T055` iteration summary integration for feedback signals
- `T056` final recommendation integration for gating and feedback
- `T057` headers/docstrings compliance updates

## 2. Baseline Already Completed

The following work is complete and validated:

- `T001`-`T018` (setup + foundational infrastructure)
- `T019`-`T032` (US1 iterative loop + CLI + interruption/reporting)
- `T033`-`T045` (US2 backend routing + robustness probes + risk-aware selection)

Current quality gates passing after Phase 4 work:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests\unit tests\contract tests\integration tests\e2e -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools\quality\validate_contracts.py
venv\Scripts\python.exe tools\quality\check_headers.py src\rewardlab tests tools
```

Important completion note:

- Deterministic backend coverage is in place today, but full project completion
  still requires the real-runtime validation work now tracked in Phase 6
  (`T066`, `T067`, with evidence captured by `T064`).

## 3. Module Map (Where Things Are)

Orchestration and CLI:

- `src/rewardlab/cli/app.py`
- `src/rewardlab/cli/session_commands.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/reporting.py`

Selection and robustness:

- `src/rewardlab/selection/policy.py`
- `src/rewardlab/selection/risk_analyzer.py`
- `src/rewardlab/experiments/robustness_runner.py`
- `src/rewardlab/experiments/backends/base.py`
- `src/rewardlab/experiments/backends/factory.py`
- `src/rewardlab/experiments/backends/gymnasium_backend.py`
- `src/rewardlab/experiments/backends/isaacgym_backend.py`

Persistence:

- `src/rewardlab/persistence/sqlite_store.py`
- `src/rewardlab/persistence/event_log.py`
- `src/rewardlab/persistence/session_repository.py`

Schemas:

- `src/rewardlab/schemas/session_config.py`
- `src/rewardlab/schemas/session_report.py`
- `src/rewardlab/schemas/reward_candidate.py`
- `src/rewardlab/schemas/reflection_record.py`
- `src/rewardlab/schemas/experiment_run.py`
- `src/rewardlab/schemas/robustness_assessment.py`

Current tests:

- `tests/unit/test_foundational_components.py`
- `tests/contract/test_session_lifecycle_cli.py`
- `tests/contract/test_backend_adapters.py`
- `tests/integration/test_backend_selection.py`
- `tests/integration/test_reward_hack_probes.py`
- `tests/integration/test_iteration_loop.py`
- `tests/e2e/test_interrupt_best_candidate.py`

## 4. Phase 5 Design and Execution Plan (Dependency Ordered)

### Chunk A: Lock feedback contracts first

Tasks: `T046`, `T047`, `T048`

Deliverables:

- CLI contracts for `feedback submit-human` and `feedback request-peer`.
- Integration tests for feedback gate modes and conflict handling.
- Deterministic fixtures that do not require live external reviewer calls.

Files to add:

- `tests/contract/test_feedback_cli.py`
- `tests/integration/test_feedback_gating.py`
- `tests/integration/test_feedback_conflicts.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_feedback_cli.py tests/integration/test_feedback_gating.py tests/integration/test_feedback_conflicts.py -q
```

### Chunk B: Add feedback data model before services

Tasks: `T049`

Deliverables:

- `FeedbackEntry` schema aligned with `data-model.md`.
- Validation for required comments and optional artifact refs.

Files to add:

- `src/rewardlab/schemas/feedback_entry.py`

Suggested gate:

```powershell
venv\Scripts\python.exe -m pytest tests/unit/test_foundational_components.py -q
```

### Chunk C: Implement feedback services and artifact tracking

Tasks: `T050`, `T051`, `T053`

Deliverables:

- Human feedback ingestion service using persisted session/candidate context.
- Peer feedback client that keeps review context isolated from orchestration state.
- Demonstration artifact tracker for linking visual evidence to feedback.

Files to add:

- `src/rewardlab/feedback/human_feedback_service.py`
- `src/rewardlab/feedback/peer_feedback_client.py`
- `src/rewardlab/feedback/demo_artifacts.py`

Integration constraints:

- Keep peer review deterministic in tests; avoid live API requirements.
- Preserve the current isolated-context design from `research.md`.

### Chunk D: Add gating evaluator and CLI wiring

Tasks: `T052`, `T054`

Deliverables:

- Feedback gate evaluator for `none`, `one_required`, and `both_required`.
- CLI feedback command handlers registered alongside the existing session app.

Files to add/update:

- `src/rewardlab/feedback/gating.py`
- `src/rewardlab/cli/feedback_commands.py`
- `src/rewardlab/cli/app.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_feedback_cli.py tests/integration/test_feedback_gating.py -q
```

### Chunk E: Integrate feedback into iteration and final recommendation paths

Tasks: `T055`, `T056`, `T057`

Deliverables:

- Feedback signals surfaced in iteration summaries and report exports.
- Final recommendation path respects gating requirements before selection is finalized.
- Policy bonuses for human and peer feedback wired through `CandidateSignal`.

Files to add/update:

- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/selection/policy.py`
- `src/rewardlab/orchestrator/reporting.py`
- `src/rewardlab/feedback/human_feedback_service.py`
- `src/rewardlab/feedback/peer_feedback_client.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/integration/test_feedback_gating.py tests/integration/test_feedback_conflicts.py tests/integration/test_iteration_loop.py -q
venv\Scripts\python.exe tools/quality/check_headers.py src/rewardlab tests tools
```

## 5. Important Integration Notes

- `SessionConfig.feedback_gate` already exists and should drive the new gating logic.
- `CandidateSignal` already exposes `human_feedback_bonus` and `peer_feedback_bonus`; Phase 5 should populate those instead of introducing parallel scoring fields.
- `session_service.py` currently stores robustness assessments and selection summaries in session metadata. Reuse that pattern or move to a more explicit persistence shape consistently.
- `app.py` currently only mounts `session_app`; feedback commands need to be registered there.

## 6. Required Regression Guardrail

Re-run existing lifecycle and robustness tests after any shared orchestration or policy change:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_session_lifecycle_cli.py tests/contract/test_backend_adapters.py tests/integration/test_backend_selection.py tests/integration/test_reward_hack_probes.py tests/integration/test_iteration_loop.py tests/e2e/test_interrupt_best_candidate.py -q -p no:cacheprovider -p no:tmpdir
```

Goal: no regression in session lifecycle, backend routing, robustness reporting, or interruption behavior.

## 7. Completion Checklist for Phase 5

Before marking US3 done:

- All tasks `T046`-`T057` checked complete in `tasks.md`
- Feedback CLI contracts and gating/conflict integration tests are green
- New/updated modules include required headers with updated dates
- New/updated public functions and methods have docstrings
- Dead-code cleanup pass completed
- `README.md` and `quickstart.md` updated if operator flows or outputs changed
- Phase 6 real-runtime validation remains queued; do not treat US3 completion as
  equivalent to full project completion

## 8. Next Agent Start Steps

From a fresh thread:

1. Checkout `iterative-main`.
2. Create a new feature branch from `iterative-main`.
3. Run the baseline verification commands from Section 2.
4. Execute chunks A-E in order with commit boundaries at chunk completion.
5. Re-run full quality gates before handing off.
