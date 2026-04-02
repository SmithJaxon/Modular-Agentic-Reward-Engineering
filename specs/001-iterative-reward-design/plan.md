# Implementation Plan: LLM-Guided Reward Function Iteration

**Branch**: `001-iterative-reward-design` | **Date**: 2026-04-02 | **Spec**: `/specs/001-iterative-reward-design/spec.md`
**Input**: Feature specification from `/specs/001-iterative-reward-design/spec.md`

## Summary

Build a modular Python research pipeline that iteratively designs and validates
reward functions for reinforcement learning environments using LLM-guided
reflection, robustness testing, and optional human/peer feedback. The system is
designed around small testable components and explicit verification loops so each
phase can be validated independently before full-session runs.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: PyTorch, Gymnasium, Isaac Gym, OpenAI API client, Pydantic, Typer
**Storage**: Local artifacts (`JSONL`, `JSON`, rendered media) and lightweight SQLite metadata
**Testing**: pytest, pytest-xdist, hypothesis, pytest-cov, mypy, ruff
**Target Platform**: Linux or macOS workstation, optional CUDA GPU
**Project Type**: Modular Python CLI application for RL experiment orchestration
**Performance Goals**: Complete a full iteration cycle in <= 120 seconds in local baseline environments; produce interruption-safe best-candidate export in <= 60 seconds
**Constraints**: API keys only via environment variables, isolated orchestration vs experiment context, resumable paused sessions, bounded retry/backoff, explicit selection rationale logging, backend chosen per session (`gymnasium` or `isaacgym`)
**Scale/Scope**: One active optimization session per environment in v1; backend selected per session; iteration limits fully user-provided

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design and before implementation.*

- PASS: Python implementation scope is explicit and aligned with project standards.
- PASS: Module boundaries and responsibilities are defined for all planned code changes.
- PASS: File header plan is documented for all new and modified Python files:
  summary, created date, and last updated date in `YYYY-MM-DD` format.
- PASS: Function and method header plan is documented for all new and modified routines.
- PASS: A dead-code cleanup pass is scheduled before merge with an owner and verification step.
- PASS: No constitution exception is required at planning time.

## Project Structure

### Documentation (this feature)

```text
specs/001-iterative-reward-design/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- orchestrator-cli.md
|   |-- session-config.schema.json
|   `-- session-report.schema.json
`-- tasks.md
```

### Source Code (repository root)

```text
src/
`-- rewardlab/
    |-- cli/
    |-- orchestrator/
    |-- llm/
    |-- experiments/
    |   `-- backends/
    |-- feedback/
    |-- selection/
    |-- persistence/
    |-- schemas/
    `-- utils/

tools/
|-- fixtures/
|-- reward_hack_probes/
`-- quality/

tests/
|-- unit/
|-- integration/
|-- contract/
`-- e2e/
```

**Structure Decision**: Use a single Python project with strict module boundaries.
Each major decision domain (orchestration, experimentation, feedback, selection,
persistence) is independently testable with dedicated contract and integration tests.

## Phase 0 Research Output

Research consolidated in `/specs/001-iterative-reward-design/research.md` with
explicit decisions on model stack, experiment strategy, robustness checks,
feedback channels, and verification tooling.

## Phase 1 Design Output

- Data model: `/specs/001-iterative-reward-design/data-model.md`
- Interface contracts: `/specs/001-iterative-reward-design/contracts/`
- Validation workflow: `/specs/001-iterative-reward-design/quickstart.md`

## Post-Design Constitution Check

- PASS: Design preserves modular decomposition with clear ownership boundaries.
- PASS: Contracts and data model define testable interfaces before implementation.
- PASS: Header requirements are reflected in quickstart quality gates.
- PASS: Dead-code cleanup is included as a mandatory pre-merge verification step.

## Complexity Tracking

No constitution violations or justified exceptions at planning time.
