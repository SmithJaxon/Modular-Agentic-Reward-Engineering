# Phase 6 Handoff: Polish, Validation, and Completion Evidence

**Prepared**: 2026-04-02  
**Feature**: `001-iterative-reward-design`  
**Prepared from implementation branch**: `001-iterative-reward-design-impl-p5`  
**Target integration branch for next work**: `iterative-main`

## 1. Scope for the Next Thread

Implement Phase 6 from `tasks.md`:

- `T058` update operator documentation and examples
- `T059` validation runner script
- `T060` end-to-end cycle-time budget test
- `T061` schema contract regression tests
- `T062` header/docstring audit tests
- `T063` dead-code cleanup pass
- `T064` full quality gate execution and verification evidence capture
- `T065` completion notes and compliance status updates
- `T066` real Gymnasium runtime smoke/integration validation
- `T067` real Isaac Gym runtime smoke/integration validation

## 2. Baseline Already Completed

The following work is complete and validated:

- `T001`-`T018` (setup + foundational infrastructure)
- `T019`-`T032` (US1 iterative loop + CLI + interruption/reporting)
- `T033`-`T045` (US2 backend routing + robustness probes + risk-aware selection)
- `T046`-`T057` (US3 human feedback, peer feedback, gating, and feedback-aware reporting)

Current quality gates passing after Phase 5 work:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests\unit tests\contract tests\integration tests\e2e -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools\quality\validate_contracts.py
venv\Scripts\python.exe tools\quality\check_headers.py src\rewardlab tests tools
```

Important completion note:

- Full project completion remains blocked on Phase 6 runtime validation evidence.
- `T066` and `T067` require supported local environments; do not mark the project
  complete until their results are captured by `T064`.

## 3. Module Map (Where Things Are)

Operator docs and artifacts:

- `README.md`
- `specs/001-iterative-reward-design/quickstart.md`
- `specs/001-iterative-reward-design/tasks.md`
- `specs/001-iterative-reward-design/checklists/requirements.md`

CLI and orchestration:

- `src/rewardlab/cli/app.py`
- `src/rewardlab/cli/session_commands.py`
- `src/rewardlab/cli/feedback_commands.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/reporting.py`

Validation and quality tooling:

- `tools/quality/validate_contracts.py`
- `tools/quality/check_headers.py`
- `tools/reward_hack_probes/probe_matrix.yaml`

Current tests:

- `tests/unit/test_foundational_components.py`
- `tests/contract/test_session_lifecycle_cli.py`
- `tests/contract/test_backend_adapters.py`
- `tests/contract/test_feedback_cli.py`
- `tests/integration/test_backend_selection.py`
- `tests/integration/test_reward_hack_probes.py`
- `tests/integration/test_iteration_loop.py`
- `tests/integration/test_feedback_gating.py`
- `tests/integration/test_feedback_conflicts.py`
- `tests/e2e/test_interrupt_best_candidate.py`

## 4. Phase 6 Design and Execution Plan (Dependency Ordered)

### Chunk A: Finalize operator docs and validation entrypoint

Tasks: `T058`, `T059`

Deliverables:

- Operator docs align with current CLI/report behavior and validation flow.
- One-command validation script separates deterministic checks from
  environment-gated real-runtime checks.

Files to add/update:

- `README.md`
- `specs/001-iterative-reward-design/quickstart.md`
- `tools/quality/run_full_validation.ps1`

Gate:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -DeterministicOnly
```

### Chunk B: Add missing regression tests around budgets and audits

Tasks: `T060`, `T061`, `T062`

Deliverables:

- Cycle-time budget test for the deterministic orchestration path.
- Schema contract regression coverage.
- Automated tests for header/docstring audit behavior.

Files to add:

- `tests/e2e/test_iteration_cycle_budget.py`
- `tests/unit/test_schema_contracts.py`
- `tests/unit/test_header_audit.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/unit/test_schema_contracts.py tests/unit/test_header_audit.py tests/e2e/test_iteration_cycle_budget.py -q -p no:cacheprovider -p no:tmpdir
```

### Chunk C: Cleanup pass and evidence generation

Tasks: `T063`, `T064`, `T065`

Deliverables:

- Dead-code cleanup completed across `src/rewardlab/` and `tests/`.
- Verification evidence captured in `verification-report.md`.
- Completion checklist updated with deterministic and runtime validation status.

Files to add/update:

- `specs/001-iterative-reward-design/verification-report.md`
- `specs/001-iterative-reward-design/checklists/requirements.md`

Integration constraints:

- Do not delete user-authored artifacts unless they are clearly stale and scoped
  to this feature branch.
- Keep deterministic validation green before attempting runtime suites.

### Chunk D: Real runtime validation on supported machines

Tasks: `T066`, `T067`

Deliverables:

- Gymnasium runtime smoke/integration coverage.
- Isaac Gym runtime smoke/integration coverage with explicit skip or gating when
  the environment is unavailable.

Files to add:

- `tests/integration/test_gymnasium_runtime.py`
- `tests/integration/test_isaacgym_runtime.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/integration/test_gymnasium_runtime.py tests/integration/test_isaacgym_runtime.py -q -p no:cacheprovider -p no:tmpdir
```

## 5. Important Integration Notes

- `session report` now includes feedback counts and gate-aware recommendation
  summaries; keep docs and examples consistent with that behavior.
- `SessionReport.status` now supports `running` exports because reports can be
  emitted before a session reaches a terminal state.
- `SessionService.step_session()` is backward-compatible with older iteration
  engine test doubles that do not accept the new `feedback_summary` parameter.
- Runtime validation should be environment-gated and must not make the
  deterministic suite flaky on machines without Gymnasium or Isaac Gym.

## 6. Required Regression Guardrail

Before and after runtime-test work, re-run the deterministic baseline:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests\unit tests\contract tests\integration tests\e2e -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools\quality\validate_contracts.py
venv\Scripts\python.exe tools\quality\check_headers.py src\rewardlab tests tools
```

Goal: preserve lifecycle, robustness, feedback, and interruption behavior while
adding validation depth.

## 7. Completion Checklist for Phase 6

Before marking the feature complete:

- All tasks `T058`-`T067` checked complete in `tasks.md`
- `run_full_validation.ps1` exists and documents deterministic vs runtime modes
- Deterministic quality gates are green
- Real Gymnasium and Isaac Gym runtime evidence is captured or explicitly gated
  with supported-machine notes
- `verification-report.md` records commands, dates, environment assumptions, and results
- `checklists/requirements.md` reflects final completion status
- Dead-code cleanup pass completed and verified

## 8. Next Agent Start Steps

From a fresh thread:

1. Checkout `iterative-main`.
2. Create a new feature branch from `iterative-main`.
3. Run the baseline deterministic verification commands from Section 2.
4. Execute Phase 6 chunks A-D in order.
5. Re-run full quality gates and capture evidence before handing off or closing the feature.
