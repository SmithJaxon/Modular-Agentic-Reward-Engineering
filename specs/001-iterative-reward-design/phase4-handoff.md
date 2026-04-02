# Phase 4 Handoff: Reward Hacking and Overfitting Detection

**Prepared**: 2026-04-02  
**Feature**: `001-iterative-reward-design`  
**Current implementation branch**: `001-iterative-reward-design-impl-p1-p3`  
**Target integration branch for next work**: `iterative-main`

## 1. Scope for the Next Thread

Implement Phase 4 / US2 from `tasks.md`:

- `T033` backend adapter contract tests
- `T034` reward-hack probe matrix integration test
- `T035` backend selection routing integration test
- `T036` Gymnasium backend adapter
- `T037` Isaac Gym backend adapter
- `T038` backend adapter factory
- `T039` experiment run schema
- `T040` robustness runner
- `T041` robustness assessment schema and risk summary
- `T042` risk analyzer and rationale capture
- `T043` probe matrix config
- `T044` policy integration of robustness outputs
- `T045` headers/docstrings compliance updates

## 2. Baseline Already Completed

The following work is complete and validated:

- `T001`-`T018` (setup + foundational infrastructure)
- `T019`-`T032` (US1 iterative loop + CLI + interruption/reporting)

Reference commits:

- `3bcd179` setup/foundation (`T001`-`T018`)
- `069a381` US1 implementation (`T019`-`T032`)

Current tests already passing before Phase 4 work:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests/unit/test_foundational_components.py tests/contract/test_session_lifecycle_cli.py tests/integration/test_iteration_loop.py tests/e2e/test_interrupt_best_candidate.py -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools/quality/validate_contracts.py
venv\Scripts\python.exe tools/quality/check_headers.py src/rewardlab tests tools
```

## 3. Module Map (Where Things Are)

Orchestration and CLI:

- `src/rewardlab/cli/app.py`
- `src/rewardlab/cli/session_commands.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/checkpointing.py`
- `src/rewardlab/orchestrator/reporting.py`

Persistence:

- `src/rewardlab/persistence/sqlite_store.py`
- `src/rewardlab/persistence/event_log.py`
- `src/rewardlab/persistence/session_repository.py`

Schemas and selection:

- `src/rewardlab/schemas/session_config.py`
- `src/rewardlab/schemas/reward_candidate.py`
- `src/rewardlab/schemas/reflection_record.py`
- `src/rewardlab/schemas/session_report.py`
- `src/rewardlab/selection/policy.py`

Experiment backend foundation:

- `src/rewardlab/experiments/backends/base.py`

Current tests:

- `tests/unit/test_foundational_components.py`
- `tests/contract/test_session_lifecycle_cli.py`
- `tests/integration/test_iteration_loop.py`
- `tests/e2e/test_interrupt_best_candidate.py`

Quality tooling:

- `tools/quality/validate_contracts.py`
- `tools/quality/check_headers.py`

## 4. Phase 4 Design and Execution Plan (Dependency Ordered)

### Chunk A: Lock test contracts first

Tasks: `T033`, `T035`

Deliverables:

- Define adapter behavior contract for both backends without requiring real GPU environments.
- Verify `environment_backend` routes to the correct adapter through factory logic.

Files to add:

- `tests/contract/test_backend_adapters.py`
- `tests/integration/test_backend_selection.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_backend_adapters.py tests/integration/test_backend_selection.py -q
```

### Chunk B: Add schemas before behavior

Tasks: `T039`, `T041`

Deliverables:

- `ExperimentRun` schema for primary and probe experiment records.
- `RobustnessAssessment` schema with explicit risk classification and summary fields.
- Preserve serialization compatibility with current report and repository patterns.

Files to add:

- `src/rewardlab/schemas/experiment_run.py`
- `src/rewardlab/schemas/robustness_assessment.py`

Suggested gate:

```powershell
venv\Scripts\python.exe -m pytest tests/unit/test_foundational_components.py -q
```

### Chunk C: Backend adapters and factory

Tasks: `T036`, `T037`, `T038`

Deliverables:

- `gymnasium_backend.py` adapter implementing `run_performance` and `run_reflection`.
- `isaacgym_backend.py` adapter implementing the same interface.
- `factory.py` adapter resolver keyed by `EnvironmentBackend`.

Files to add:

- `src/rewardlab/experiments/backends/gymnasium_backend.py`
- `src/rewardlab/experiments/backends/isaacgym_backend.py`
- `src/rewardlab/experiments/backends/factory.py`

Integration constraints:

- Keep tests deterministic; support mock/fixture-friendly execution paths.
- Use lazy imports for optional backends so test runs do not hard-fail when Isaac Gym is unavailable.

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_backend_adapters.py tests/integration/test_backend_selection.py -q
```

### Chunk D: Probe matrix and robustness runner

Tasks: `T043`, `T040`, `T034`

Deliverables:

- Probe matrix config that varies architecture/hyperparameters.
- Runner that executes probe variants for a candidate and aggregates outcomes.
- Integration test validating reward-hacking risk behavior from probe outcomes.

Files to add:

- `tools/reward_hack_probes/probe_matrix.yaml`
- `src/rewardlab/experiments/robustness_runner.py`
- `tests/integration/test_reward_hack_probes.py`

Gate:

```powershell
venv\Scripts\python.exe -m pytest tests/integration/test_reward_hack_probes.py -q
```

### Chunk E: Risk analyzer and policy integration

Tasks: `T042`, `T044`, `T045`

Deliverables:

- Analyzer that converts probe outputs into risk signals and textual rationale.
- Policy integration that balances performance and risk without hard-threshold rigidity.
- Explicit rationale capture when selecting a candidate with known minor robustness risk.

Files to add/update:

- `src/rewardlab/selection/risk_analyzer.py` (new)
- `src/rewardlab/selection/policy.py` (update)
- `src/rewardlab/orchestrator/session_service.py` or report shaping path (if required for rationale surfacing)
- `src/rewardlab/experiments/robustness_runner.py` (header/docstring pass)

Gates:

```powershell
venv\Scripts\python.exe -m pytest tests/integration/test_reward_hack_probes.py tests/integration/test_iteration_loop.py -q
venv\Scripts\python.exe tools/quality/check_headers.py src/rewardlab tests tools
```

## 5. Required Regression Guardrail

Run existing US1 tests after each Phase 4 chunk that touches shared orchestration files:

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_session_lifecycle_cli.py tests/integration/test_iteration_loop.py tests/e2e/test_interrupt_best_candidate.py -q -p no:cacheprovider -p no:tmpdir
```

Goal: no regression in session lifecycle, interruption handling, or report export.

## 6. Runtime and Budget Constraints

- Keep all installs and execution inside `venv`.
- Do not require real OpenAI API calls for Phase 4 tests.
- Treat Gymnasium and Isaac Gym as selectable backends, but keep tests runnable without machine-specific Isaac Gym setup.
- Avoid introducing global or machine-level dependencies outside this repository.

## 7. Completion Checklist for Phase 4

Before marking US2 done:

- All tasks `T033`-`T045` checked complete in `tasks.md`
- New/updated modules include required headers with updated dates
- New/updated public functions and methods have docstrings
- Dead-code cleanup pass completed
- Phase 4 verification commands recorded in a status note (or commit message body)
- `quickstart.md` updated if any operator commands changed

## 8. Next Agent Start Steps

From a fresh thread:

1. Checkout `iterative-main`.
2. Create a new feature branch from `iterative-main`.
3. Run baseline verification commands from Section 2.
4. Execute chunks A-E in order with commit boundaries at chunk completion.
5. Re-run full quality gates before handing off.
