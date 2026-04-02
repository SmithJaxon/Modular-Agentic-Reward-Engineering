# RewardLab: LLM-Guided Reward Function Iteration

This repository contains a modular Python pipeline for iteratively designing and
validating reinforcement-learning reward functions using LLM-guided reflection,
robustness checks, and human or peer feedback gates.

## Current Status (2026-04-02)

- Implemented: Phase 1 (setup), Phase 2 (foundational infra), Phase 3 (US1/MVP loop)
- Next target: Phase 4 (US2 reward-hacking and overfitting detection)
- Active implementation branch for completed work: `001-iterative-reward-design-impl-p1-p3`

## Primary Docs

- Feature spec: `specs/001-iterative-reward-design/spec.md`
- Implementation plan: `specs/001-iterative-reward-design/plan.md`
- Task board: `specs/001-iterative-reward-design/tasks.md`
- Phase 4 handoff and execution plan: `specs/001-iterative-reward-design/phase4-handoff.md`
- Operator quickstart: `specs/001-iterative-reward-design/quickstart.md`

## Local Setup (Project-Scoped Venv)

```powershell
venv\Scripts\python.exe -m pip install -e .[dev]
```

## Validation Commands

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests\unit\test_foundational_components.py tests\contract\test_session_lifecycle_cli.py tests\integration\test_iteration_loop.py tests\e2e\test_interrupt_best_candidate.py -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools\quality\validate_contracts.py
venv\Scripts\python.exe tools\quality\check_headers.py src\rewardlab tests tools
```

## Runtime Artifacts

By default the CLI writes session state under `.rewardlab/`:

- `rewardlab.sqlite3` for metadata
- `events/*.jsonl` for append-only event logs
- `checkpoints/*.json` for interruption/resume snapshots
- `reports/*.json` for exported session reports

## Engineering Standards

Standards are defined in `.specify/memory/constitution.md`, including modularity,
required module headers, required routine docstrings, and dead-code cleanup
before merge.
