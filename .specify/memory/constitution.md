<!--
Sync Impact Report
- Version change: unversioned template -> 1.0.0
- Modified principles:
  - PRINCIPLE_1_NAME -> I. Modular Architecture by Default
  - PRINCIPLE_2_NAME -> II. Mandatory File Header Blocks
  - PRINCIPLE_3_NAME -> III. Mandatory Function and Method Header Blocks
  - PRINCIPLE_4_NAME -> IV. Python Consistency and Readability Rules
  - PRINCIPLE_5_NAME -> V. Dead Code Elimination Before Merge
- Added sections:
  - Implementation Standards
  - Review Workflow and Quality Gates
- Removed sections:
  - None
- Templates requiring updates:
  - UPDATED: .specify/templates/plan-template.md
  - UPDATED: .specify/templates/spec-template.md
  - UPDATED: .specify/templates/tasks-template.md
  - PENDING: .specify/templates/commands/*.md (directory not present in repository)
  - UPDATED: README.md
- Follow-up TODOs:
  - TODO(COMMAND_TEMPLATES): Apply these constitution gates when command templates are added.
-->
# Advanced AI Project Constitution

## Core Principles

### I. Modular Architecture by Default
All Python code MUST be organized into small, focused modules with clear boundaries
and explicit responsibilities. New features MUST be implemented by extending
existing modules or introducing new modules with single, well-defined purposes.
Cross-module coupling MUST be minimized through explicit interfaces.

Rationale: modular design keeps complexity manageable, improves reuse, and reduces
regression risk when requirements evolve.

### II. Mandatory File Header Blocks
Every Python source file MUST begin with a header block (module docstring) that
includes: a concise file summary, `Created` date, and `Last Updated` date in
ISO format (`YYYY-MM-DD`). The `Last Updated` field MUST be changed whenever the
file logic changes.

Rationale: consistent headers improve maintainability, accelerate onboarding, and
provide visible change context during review.

### III. Mandatory Function and Method Header Blocks
Every function and method MUST include a header block (docstring). Each header
MUST describe purpose and behavior; for non-trivial logic it MUST also describe
inputs, outputs, raised exceptions, and side effects. Public APIs MUST include
type hints and docstrings that stay aligned with implementation changes.

Rationale: predictable method-level documentation reduces ambiguity and makes code
safer to modify.

### IV. Python Consistency and Readability Rules
The project MUST use Python as the implementation language and MUST follow
consistent Python conventions across all modules: descriptive naming, explicit
imports, formatting/linting compliance, and readable control flow. New and changed
public interfaces MUST use type hints. Complex logic MUST be decomposed into
smaller functions instead of deeply nested blocks.

Rationale: consistent style lowers cognitive load and enables reliable review and
tooling.

### V. Dead Code Elimination Before Merge
Every feature branch MUST include at least one explicit cleanup pass before merge.
This pass MUST remove dead code, unreachable branches, stale imports, unused
variables, obsolete comments, and abandoned feature toggles unless an active
tracking issue justifies retention.

Rationale: ongoing cleanup prevents codebase entropy and keeps implementation
intent clear.

## Implementation Standards

- Header block template for Python files:
  - `Summary: <one or two sentences>`
  - `Created: <YYYY-MM-DD>`
  - `Last Updated: <YYYY-MM-DD>`
- Function and method headers MUST use docstrings. Single-line docstrings are
  allowed only for truly trivial behavior; all other routines require structured
  detail.
- File and method headers are required for all new files and all modified
  functions/methods in touched files.
- Pull requests MUST include evidence of cleanup pass completion when code is
  added or changed.

## Review Workflow and Quality Gates

- Plan Gate: implementation plans MUST include a constitution check that validates
  modular decomposition, file headers, method headers, and cleanup activities.
- Spec Gate: feature specs MUST capture any scope that materially affects module
  boundaries or documentation obligations.
- Task Gate: task lists MUST include explicit tasks for documentation headers and
  dead code cleanup where code changes are present.
- Review Gate: reviewers MUST block merge when constitution requirements are
  missing, outdated, or unverifiable.

## Governance

This constitution is the highest-priority engineering policy for this repository.
When conflicts occur, this document takes precedence over ad hoc conventions.

Amendment procedure:
1. Propose changes in a pull request that includes rationale and migration impact.
2. Obtain approval from maintainers responsible for code quality standards.
3. Update dependent templates and guidance files in the same change.

Versioning policy:
- MAJOR: backward-incompatible governance changes or principle removals/redefinitions.
- MINOR: new principle/section or materially expanded guidance.
- PATCH: clarifications, wording improvements, and non-semantic refinements.

Compliance review expectations:
- Every feature plan, specification, task list, and code review MUST include an
  explicit constitution compliance check.
- Non-compliant pull requests MUST be corrected before merge.

**Version**: 1.0.0 | **Ratified**: 2026-04-02 | **Last Amended**: 2026-04-02
