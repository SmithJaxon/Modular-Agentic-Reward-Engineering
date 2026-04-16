# Implementation Plan: Autonomous Project Completion Pass

**Branch**: `agent-autonomous-pass` | **Date**: 2026-04-02 | **Spec**: [spec.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\002-autonomous-project-pass\spec.md)
**Input**: Operating specification from `specs/002-autonomous-project-pass/spec.md`

## Summary

Run the full `001-iterative-reward-design` backlog from the dedicated worktree in
one uninterrupted pass whenever local execution allows it. Work proceeds in
dependency order, uses parallel sub-agents for disjoint chunks, validates every
chunk before advancing, and only asks the user for approval when a hard gate is
triggered.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, Gymnasium, OpenAI API client, Pydantic, Typer  
**Storage**: Local JSONL, JSON, rendered media, lightweight SQLite metadata  
**Testing**: pytest, targeted contract/integration/e2e coverage, ruff  
**Target Platform**: Local Windows worktree driven from PowerShell  
**Project Type**: Python CLI and orchestration toolkit  
**Performance Goals**: Complete each chunk with the smallest reliable validation set before escalation to broader checks  
**Constraints**: No downloads without user approval, no global installs, no changes outside the worktree, minimal API spend, no paid API execution before `.env` credentials are supplied  
**Scale/Scope**: Entire `specs/001-iterative-reward-design/tasks.md` backlog

## Constitution Check

*GATE: This operating plan must hold for the entire autonomous pass.*

- Python implementation scope remains explicit and aligned with project standards.
- Module boundaries follow the existing `src/rewardlab/` package layout and
  task-level file ownership.
- New and modified Python files must receive required module headers.
- New and modified functions and methods must receive aligned docstrings.
- A dead-code cleanup and compliance pass is reserved for the final project stage.
- No constitution exception is planned in this pass.

## Project Structure

### Documentation (autonomous pass)

```text
specs/002-autonomous-project-pass/
|-- spec.md
|-- plan.md
`-- checklists/
    `-- requirements.md
```

### Source Code (execution target)

```text
src/rewardlab/
|-- cli/
|-- experiments/
|   `-- backends/
|-- feedback/
|-- llm/
|-- orchestrator/
|-- persistence/
|-- schemas/
|-- selection/
`-- utils/

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

**Structure Decision**: The pass will implement the existing single-project Python
layout already defined by the `001-iterative-reward-design` plan and tasks.

## Execution Protocol

### Worktree Boundary

- Active worktree: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- All code edits, tests, temporary environments, cached artifacts, and generated
  evidence stay inside the active worktree.
- No commands may alter machine-global configuration, user profiles, registry,
  PATH, or sibling directories without explicit user approval.

### Authority Order

1. User instructions in this thread
2. `AGENTS.md`
3. `specs/002-autonomous-project-pass/spec.md`
4. `specs/001-iterative-reward-design/tasks.md`
5. Existing feature design artifacts under `specs/001-iterative-reward-design/`

### Chunk Loop

For each eligible chunk or safe task group:

1. Confirm dependency readiness from `tasks.md`.
2. Decide whether the work can be parallelized with disjoint ownership.
3. Create or update tests first when coverage is missing or stale.
4. Implement or revise code in the smallest coherent slice.
5. Run the narrowest validation set that can prove the slice is correct.
6. Fix failures immediately and rerun validation until the slice is stable.
7. Record evidence, update completion state, and advance to the next chunk.

### Phase Order

1. Phase 1 setup tasks `T001-T008`
2. Phase 2 foundational tasks `T009-T018`
3. Phase 3 user story 1 tasks `T019-T032`
4. Phase 4 user story 2 tasks `T033-T045`
5. Phase 5 user story 3 tasks `T046-T057`
6. Phase 6 polish tasks `T058-T065`

### Parallelization Rules

- Parallelize only when tasks have no unmet dependency edge and their write sets
  are disjoint.
- Prefer sub-agents for sidecar work such as isolated tests, independent schemas,
  backend adapters, or documentation files.
- Additional local worktrees may be used for larger independent streams when branch
  isolation is cleaner than patch-level integration.
- Keep orchestration, shared interfaces, shared persistence code, and final
  verification under the main agent.
- Re-run integration checks after merging any parallel work streams.

### Commit And Review Cadence

- Commit early and often at coherent chunk boundaries with meaningful commit
  messages tied to the work actually completed.
- Before accepting sub-agent code, review it as a pull request: read the diff,
  challenge weak assumptions, confirm tests, and make follow-up edits if needed.
- Before merging extra-worktree branches, perform the same pull-request style
  review and validate the merged result in the main worktree.
- Use meaningful merge or squash commit messages for integrated parallel work so
  the branch history remains interpretable.

### Validation Ladder

Use the cheapest proof that gives confidence, escalating only when needed:

1. Static inspection and focused unit tests
2. Contract tests for touched interfaces
3. Targeted integration tests for the current chunk
4. End-to-end or full quality gate only when chunk integration requires it

### Approval Gates

The user must be asked before:

- Any download, install, upgrade, package fetch, model pull, or dataset fetch
- Any machine-level change or path outside the worktree
- Any paid API call when the needed credential is absent
- Any destructive action with irreversible data-loss risk
- Any decision with no reasonable default that would materially change contract
  behavior

### Dependency and Credential Policy

- Approved downloads must target a worktree-local `.venv\` or other project-local
  directory only.
- No global package installs or environment mutation are allowed.
- If API-backed validation is required, prompt the user for the `.env` key at that
  moment and do not spend budget earlier.
- Paid API smoke tests should use the cheapest viable model and the fewest calls
  possible.

### Blocker Policy

- When a gated dependency blocks only part of the backlog, continue all other
  dependency-safe work first.
- When unexpected edits appear in active files, stop and ask the user before
  continuing in those files.
- When a blocker remains at the end of all other work, report the exact blocked
  tasks, missing prerequisite, and next command or approval needed.

### Completion Evidence

Each completed chunk should leave:

- Updated code and tests
- Validation results summarized in agent output or project evidence files
- A meaningful commit once the chunk is stable
- Updated task/checklist state where appropriate
- Remaining risk notes only when further work is genuinely blocked

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
