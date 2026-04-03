# Next Agent Handoff

## Objective

Continue from the dedicated worktree until RewardLab can run an actual
experiment in both `gymnasium` and `isaacgym`, not just the offline-safe MVP
loop.

## Where To Work

- Active worktree: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- Active branch: `agent-autonomous-pass`

Do not work in the sibling root repository for this thread.

## Primary Backlog

1. `specs/003-real-experiment-readiness/spec.md`
2. `specs/003-real-experiment-readiness/plan.md`
3. `specs/003-real-experiment-readiness/tasks.md`

`specs/001-iterative-reward-design/` is complete for the MVP/offline-safe
tranche. The remaining product gap is real backend execution, real robustness
evidence, and real artifacts.

## Current State

- Offline MVP orchestration is complete and previously validated.
- Live low-cost peer feedback path is working with `.env` autoload.
- `gymnasium` and `torch` were not installed in `.venv` at the time this
  handoff was written.
- The main remaining gap is that `session step` still uses the deterministic
  iteration engine instead of a real backend execution path.

## Hard Rules

- Keep all edits, tests, temp files, installs, and generated artifacts inside
  this worktree.
- No downloads, installs, upgrades, package fetches, model pulls, or dataset
  fetches without user approval.
- Any approved install must stay inside `.venv`.
- No machine-level changes, PATH edits, registry edits, shell profile edits, or
  writes outside the worktree.
- Make frequent meaningful commits.
- Review sub-agent code like a pull request before integration.
- Parallelize aggressively only when write sets are disjoint.
- Use the cheapest viable API model and as few calls as possible if API-backed
  validation is needed.

## Approval Gates

Stop and ask the user before:

- any new dependency install or download
- any runtime setup that writes outside the worktree
- any destructive command
- any paid API action if the current path truly requires it

If blocked on an approval-gated install, continue unblocked code and test work
first wherever dependency order allows it.

## Execution Loop

For each chunk:

1. confirm task scope from `specs/003-real-experiment-readiness/tasks.md`
2. add or update tests first
3. implement the smallest coherent slice
4. run the smallest relevant validation set
5. fix failures immediately
6. commit when the slice is stable

## Suggested First Moves

1. Read `AGENTS.md`, `specs/002-autonomous-project-pass/`, and the full `003`
   planning set.
2. Start with `T001`, `T003`, `T004`, and the Phase 2 design work that does not
   require new packages.
3. Ask for approval only when the next critical step truly needs real backend
   dependencies inside `.venv`.
4. Finish Gymnasium real execution before attempting Isaac integration.

## Completion Standard

Do not treat the project as complete until you can point to:

- one real Gymnasium session report with non-empty run metrics and artifact refs
- one real Isaac session report with non-empty run metrics and artifact refs
- the offline suite still passing
- exact approved install/setup commands recorded in docs

If you cannot finish due to an approval-gated dependency or runtime prerequisite,
leave the branch in a clean committed state and document the exact blocker,
current evidence, and next command needed.
