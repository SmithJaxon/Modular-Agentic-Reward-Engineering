# Feature Specification: Autonomous Project Completion Pass

**Feature Branch**: `agent-autonomous-pass`  
**Created**: 2026-04-02  
**Status**: Active  
**Input**: User description: "Create a new worktree for this thread, then define strict guidelines and specifications for a one-shot autonomous pass of the entire project. The agent should complete each chunk end to end with testing and iteration, parallelize aggressively across sub-agents, require user approval for downloads or any machine-level changes, ask for the API key only when needed, and keep API usage minimal and low cost."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Unattended End-to-End Delivery (Priority: P1)

The project owner wants the agent to execute the full `001-iterative-reward-design`
backlog in one pass without routine check-ins, moving through each chunk until the
entire project is complete or only explicit hard blockers remain.

**Why this priority**: This is the primary operating mode requested for the thread.
Without this, the agent would still require manual orchestration and the one-shot
delivery goal would fail.

**Independent Test**: Starting from the dedicated worktree, the agent can move
through the existing phase/task backlog, completing chunk-level test and validation
loops and advancing without user prompts unless a defined approval gate is hit.

**Acceptance Scenarios**:

1. **Given** the worktree is ready and `specs/001-iterative-reward-design/tasks.md`
   defines the backlog, **When** the agent begins execution, **Then** it proceeds in
   dependency order and does not pause for normal implementation decisions.
2. **Given** a chunk fails tests or validation, **When** the failure is detected,
   **Then** the agent iterates locally on that chunk until it passes or reaches a
   hard blocker.
3. **Given** a chunk is stable, **When** required tests and validations pass,
   **Then** the agent records the evidence and moves to the next eligible chunk.

---

### User Story 2 - Controlled Environment and Spend (Priority: P1)

The project owner wants confidence that the autonomous pass will not alter the
machine, install global software, or spend paid API budget without explicit consent.

**Why this priority**: Environment safety and cost control are non-negotiable
constraints for execution of the rest of the project.

**Independent Test**: With missing dependencies and no API key configured, the
agent can continue local work where possible, but it stops and asks the user before
any download, install, or paid external call.

**Acceptance Scenarios**:

1. **Given** a dependency is missing, **When** the agent determines that a download
   is required, **Then** it requests user approval before performing any install and
   scopes the install to a worktree-local virtual environment after approval.
2. **Given** a command would modify the machine outside the worktree, **When** the
   agent evaluates that command, **Then** it refuses to run it and surfaces the
   reason to the user.
3. **Given** API-backed validation becomes necessary and `.env` lacks a key,
   **When** the agent reaches that point, **Then** it prompts the user for the key
   and delays only the API-dependent work.

---

### User Story 3 - Parallel but Safe Execution (Priority: P2)

The project owner wants the agent to use sub-agents for speed where tasks are
independent, but still preserve deterministic integration and verification.

**Why this priority**: Parallelism is necessary to keep the one-shot pass efficient,
but uncontrolled parallel edits would increase merge risk and instability.

**Independent Test**: The agent can assign disjoint task groups to sub-agents,
integrate the results, and rerun final validation before marking a chunk complete.

**Acceptance Scenarios**:

1. **Given** multiple tasks have disjoint file ownership and no shared dependency,
   **When** the agent schedules the work, **Then** it may run those tasks in
   parallel across sub-agents.
2. **Given** tasks touch the same files or block the next integration step,
   **When** the agent plans the work, **Then** it keeps those tasks serialized.
3. **Given** sub-agent outputs are returned, **When** the main agent integrates the
   results, **Then** it reruns the relevant verification set before advancing.

---

### Edge Cases

- A required backend or library is unavailable locally and would require a new
  download before related tasks can run.
- The API key is still missing when the first paid integration test becomes
  necessary.
- A parallel chunk completes while another chunk reveals a shared-file dependency
  that invalidates the original split.
- Unexpected external edits appear in files currently being modified inside the
  worktree.
- A heavy external dependency remains blocked, but unrelated backlog items can
  still be completed locally.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All work for this thread after worktree creation MUST be performed in
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`.
- **FR-002**: The agent MUST treat
  `specs/001-iterative-reward-design/tasks.md` as the authoritative execution
  backlog for implementation chunks and dependency order.
- **FR-003**: The agent MUST use a per-chunk execution loop of test definition or
  update, implementation, targeted validation, failure repair, and re-validation
  before advancing.
- **FR-004**: The agent MUST proceed without routine user feedback and only stop
  for approval gates or irreducible blockers.
- **FR-005**: User feedback is REQUIRED before any download, install, upgrade,
  package fetch, model pull, dataset fetch, or other externally retrieved
  dependency.
- **FR-006**: Any approved dependency installation MUST remain inside a
  worktree-local environment or directory such as `.venv\`; the agent MUST NOT
  perform machine-level installs or modify PATH, registry, shell profiles, or
  other global configuration.
- **FR-007**: The agent MUST NOT modify paths outside the active worktree, except
  for git metadata operations already required to create and maintain the worktree.
- **FR-008**: The agent MUST prefer offline fixtures, mocks, schema validation, and
  local integration tests before attempting any paid or externally dependent test.
- **FR-009**: When API-backed execution is required, the agent MUST prompt the user
  to populate `.env` if the needed key is absent and MUST defer only the API-bound
  work.
- **FR-010**: API-backed smoke testing MUST use the cheapest viable model and the
  fewest calls needed to validate the integration path.
- **FR-011**: The agent MUST parallelize work across sub-agents whenever the tasks
  are independent and file ownership can be kept disjoint.
- **FR-012**: The agent MUST keep shared-file work, critical-path integration, and
  final verification serialized under the main agent.
- **FR-013**: After each completed chunk, the agent MUST record what was changed,
  which validations ran, and whether any blockers remain.
- **FR-014**: The agent MUST make frequent, meaningful commits so progress is
  captured in coherent checkpoints instead of large unreviewed batches.
- **FR-015**: The main agent MUST review sub-agent code as if it were a pull
  request before integration, including diff inspection, validation review, and
  behavioral regression checks.
- **FR-016**: The agent MAY create additional worktrees inside the workspace when
  parallel branch isolation improves safety or throughput, but each such branch
  MUST be reviewed and merged back with a meaningful merge or squash commit
  message.
- **FR-017**: If a task is blocked by an approval gate, the agent MUST continue on
  other unblocked work where dependency order allows it and MUST return later to
  the blocked task once approval is available.
- **FR-018**: If unexpected external edits appear in files being actively changed,
  the agent MUST stop and request user direction before continuing in those files.
- **FR-019**: The agent MUST keep secrets out of logs and terminal echoes and MUST
  treat `.env` contents as sensitive.

### Constitution Alignment Requirements

- **CAR-001**: The autonomous pass MUST preserve modular Python boundaries across
  `src/rewardlab/`, keeping implementation grouped by focused modules.
- **CAR-002**: The execution rules MUST explicitly cover changes to source,
  tests, tooling, and feature documentation under the existing repository
  structure so header and cleanup obligations can be planned.
- **CAR-003**: The autonomous pass MUST require header/docstring updates for every
  new or modified Python file and routine touched during implementation.
- **CAR-004**: The autonomous pass MUST include an explicit dead-code cleanup and
  final compliance review before the project is considered complete.

### Key Entities *(include if feature involves data)*

- **Worktree Environment**: The isolated repository checkout that contains all
  edits, local environments, generated artifacts, and test evidence for this
  thread.
- **Execution Chunk**: The smallest implementation unit the agent may move through
  independently, typically aligned to a task or safe task group from
  `tasks.md`.
- **Approval Gate**: A pre-defined condition that requires user input before
  execution may continue for the affected work.
- **Validation Evidence**: The tests, lint checks, schema checks, and written
  notes that show a chunk is stable before the next chunk begins.
- **API Budget Event**: A paid external call that must be minimized and justified
  relative to cheaper local validation options.
- **Reviewable Integration**: A commit, branch merge, or worktree merge that has
  been inspected by the main agent using a pull-request style review standard.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The agent can execute backlog work from the dedicated worktree
  without requesting user input except when a listed approval gate is triggered.
- **SC-002**: 100% of thread-specific edits, temporary environments, and generated
  artifacts remain inside the dedicated worktree.
- **SC-003**: 0 machine-level installs, PATH edits, registry edits, or shell
  profile changes occur during the autonomous pass.
- **SC-004**: Every completed chunk has recorded validation evidence before the
  next chunk begins.
- **SC-005**: No paid API calls occur before local validation passes and user
  credentials are supplied, and paid smoke validation uses no more than the
  smallest scenario set needed to verify the integration path.
- **SC-006**: Parallel execution is used only for disjoint work; all integrated
  changes pass the relevant post-merge verification set before chunk completion.
- **SC-007**: Progress is preserved through frequent meaningful commits, and all
  sub-agent or extra-worktree integrations receive main-agent review before merge.

## Assumptions

- The current implementation scope remains the backlog defined in
  `specs/001-iterative-reward-design/`.
- Python is already available locally so a worktree-local virtual environment can
  be created later without machine-level changes if approval for dependency
  downloads is granted.
- The user will provide the required API key in `.env` when the agent explicitly
  requests it.
- If heavyweight dependencies remain unavailable without download approval, the
  agent may complete all other independent work first and leave only the blocked
  tasks pending.
