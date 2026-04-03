# Implementation Plan: Real Experiment Readiness

**Branch**: `003-real-experiment-readiness` | **Date**: 2026-04-02 | **Spec**: [spec.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\spec.md)  
**Input**: Feature specification from `specs/003-real-experiment-readiness/spec.md`

## Summary

Extend the completed RewardLab MVP into a real backend-driven experiment system
that can execute actual experiments in both Gymnasium and Isaac Gym. The next
implementation tranche should preserve the existing offline-safe test path while
replacing the heuristic execution path for real sessions with persisted backend
run evidence, integrated robustness assessment, and real review artifacts.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: `rewardlab` package, Gymnasium, PyTorch, user-approved Isaac runtime, OpenAI client, Pydantic, Typer  
**Storage**: Worktree-local SQLite metadata plus `.rewardlab/` JSON, JSONL, and run-artifact files  
**Testing**: pytest, ruff, mypy, targeted backend smoke tests, offline regression suite  
**Target Platform**: Local Windows worktree via PowerShell, with optional GPU/runtime support for Isaac  
**Project Type**: Python CLI and orchestration toolkit for RL reward iteration  
**Performance Goals**: Real Gymnasium smoke path completes in a reproducible local budget; real Isaac smoke path completes once approved runtime prerequisites are present; offline validation remains fast enough for tight iteration  
**Constraints**: No downloads without user approval, no installs outside `.venv`, no machine-level changes, no writes outside worktree, real experiment path cannot silently fall back to heuristic scoring, API usage remains minimal and unchanged from current low-cost policy  
**Scale/Scope**: One real experiment-ready session path for Gymnasium and one for Isaac Gym, each validated through the shared session/report workflow

## Constitution Check

*GATE: Must pass before implementation begins and after the design is updated.*

- PASS: The work can be decomposed into focused modules for reward loading,
  execution coordination, artifact capture, backend adapters, and orchestration.
- PASS: New files and modified files can continue to satisfy mandatory module
  headers and function/method docstrings.
- PASS: The plan includes explicit cleanup of superseded MVP-only execution
  branches before merge.
- PASS: The design keeps heavy backend-specific details out of the session
  service by using narrower execution modules and repository helpers.
- PASS: No constitution exception is required at planning time.

## Project Structure

### Documentation (this feature)

```text
specs/003-real-experiment-readiness/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- tasks.md
`-- checklists/
    `-- requirements.md
```

### Source Code (expected touch points)

```text
src/rewardlab/
|-- cli/
|-- experiments/
|   |-- backends/
|   |-- artifacts.py
|   |-- execution_service.py
|   |-- gymnasium_runner.py
|   `-- isaacgym_runner.py
|-- feedback/
|-- orchestrator/
|-- persistence/
|-- schemas/
`-- selection/

tests/
|-- contract/
|-- e2e/
|-- integration/
`-- unit/

tools/
|-- fixtures/
|-- quality/
`-- reward_hack_probes/
```

**Structure Decision**: Keep the current modular package layout and add the real
experiment layer through narrow execution modules rather than by overloading the
existing deterministic iteration engine.

## Research Decisions

Research is recorded in [research.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\research.md).
The main decisions are:

- preserve deterministic execution only as explicit offline validation mode
- define a constrained reward-program loading interface
- persist run artifacts under `.rewardlab/` by run id
- make Gymnasium the first real backend milestone
- make Isaac readiness explicit and actionable rather than silent or fake
- validate heavy backends with opt-in smokes after approval

## Design Artifacts

- Data model: [data-model.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\data-model.md)
- Operator workflow: [quickstart.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\quickstart.md)
- Task backlog: [tasks.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\tasks.md)

## Execution Strategy

### Phase Order

1. Approval-gated setup for real backend dependencies and reproducible fixture configs
2. Foundational execution modules for reward loading, run persistence, and artifacts
3. Real Gymnasium execution integrated into the session/report lifecycle
4. Real robustness and artifact capture integrated into candidate selection and feedback
5. Real Isaac execution integrated into the same lifecycle
6. Final validation, docs, cleanup, and handoff refresh

### Validation Strategy

Use the validation ladder below so the agent can keep moving without wasting time
or budget:

1. Offline unit and contract coverage for new execution modules
2. Targeted integration coverage for Gymnasium lifecycle integration
3. Opt-in backend smoke validation after approved installs
4. Full local quality gate after each story-level integration

### Approval Strategy

- Treat dependency install and runtime acquisition as explicit user approval gates.
- Do as much code, test, and documentation work as possible before hitting the
  first blocked install step.
- After approval, keep all dependency setup inside `.venv` and document the exact
  commands and versions used.

### Parallelization Strategy

- Split work across sub-agents by module ownership:
  - execution modules and schemas
  - Gymnasium backend and lifecycle integration
  - robustness/artifact pipeline
  - Isaac backend and smoke validation
- Review each sub-agent diff like a pull request before integrating.
- Keep shared-file integration in `session_service.py`, reporting, and docs under
  the main agent.

## Post-Design Constitution Check

- PASS: Design artifacts preserve modular decomposition and make new boundaries explicit.
- PASS: The task plan includes tests, header/docstring compliance, and cleanup work.
- PASS: The plan keeps real-backend logic reviewable by separating backend-specific runners.
- PASS: Operator docs and handoff artifacts are part of the planned deliverables.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
