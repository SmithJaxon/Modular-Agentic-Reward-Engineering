# Project Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-02

## Active Technologies
- Python 3.12 + PyTorch, Gymnasium, OpenAI API client, Pydantic, Typer (001-iterative-reward-design)
- Local artifacts (`JSONL`, `JSON`, rendered media) and lightweight SQLite metadata (001-iterative-reward-design)

- Python 3.12 + PyTorch, Gymnasium, OpenAI API client, Pydantic, Typer (001-iterative-reward-design)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.12: Follow standard conventions

## Recent Changes
- 001-iterative-reward-design: Added Python 3.12 + PyTorch, Gymnasium, OpenAI API client, Pydantic, Typer

- 001-iterative-reward-design: Added Python 3.12 + PyTorch, Gymnasium, OpenAI API client, Pydantic, Typer

<!-- MANUAL ADDITIONS START -->
## Autonomous Pass Guardrails

- Active worktree for this thread: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- After worktree creation, all edits, tests, temporary files, virtual environments, and generated artifacts for this thread MUST stay inside the active worktree.
- The execution target is the active follow-on backlog for this thread. `specs/001-iterative-reward-design/tasks.md` and `specs/003-real-experiment-readiness/tasks.md` are complete in this worktree. The active implementation backlog is `specs/004-agent-tool-calling-architecture/`.
- The active runtime target remains Gymnasium-only. `CartPole-v1` is the smoke path, and `Humanoid-v4` PPO evaluation is the main real-execution target while control architecture migrates from `session` pipeline to `agent_tools`.
- For each chunk, the default loop is: confirm scope, add or update tests, implement or revise code, run the smallest relevant validation set, fix failures, rerun validation, then advance only when the chunk is stable.
- Use sub-agents aggressively for independent tasks with disjoint file ownership. Keep shared-file work, critical-path integration, and final verification in the main agent.
- The agent SHOULD make frequent, meaningful commits during execution rather than carrying large uncommitted batches for long periods.

## Commit And Review Workflow

- Commits MUST be small enough to isolate a coherent chunk, include passing relevant validation, and use meaningful commit messages that describe the actual change.
- Sub-agent output MUST be reviewed by the main agent as if it were a pull request before merge or integration: inspect the diff, verify behavior, check tests, and reject or revise weak changes.
- When parallel branches or extra worktrees materially reduce risk or integration time, the agent MAY create additional worktrees inside the workspace, complete isolated changes there, and merge them back only after review.
- Any local merge of parallel work MUST be accompanied by a meaningful merge or squash commit message that explains what was integrated and why.
- Shared-file integrations and final branch assembly remain the responsibility of the main agent after review of subordinate work.

## Hard Approval Gates

- User input is REQUIRED before any download, install, upgrade, package fetch, model pull, dataset fetch, or other network-retrieved dependency.
- Any approved dependency work MUST stay inside a worktree-local virtual environment such as `.venv\`. No machine-level installs, no global Python package changes, no PATH changes, no registry edits, no shell profile edits, and no writes outside this workspace/worktree.
- User input is REQUIRED before any command or edit that would touch paths outside the active worktree, except the already-completed git worktree creation itself.
- User input is REQUIRED when an OpenAI API key or other paid credential is needed. The agent should prompt the user to populate `.env` at that point and should not proceed with paid execution beforehand.
- User input is REQUIRED before destructive actions that could irreversibly remove user-authored data or overwrite unexpected third-party changes.

## API Cost Controls

- Prefer offline fixtures, mocks, schema checks, and local integration coverage before any paid API call.
- When API-backed validation becomes necessary, use the cheapest viable model, keep prompts minimal, and run the fewest calls needed for a smoke test.
- Do not consume API budget for broad exploratory testing when an offline substitute can validate the same behavior.

## Blocker Handling

- If a blocked task depends on an approval-gated download or credential, pause that task, record the blocker clearly, continue any other unblocked work, and return for approval only when local progress is exhausted.
- If unexpected external edits appear in files being actively changed by the agent, stop and ask the user how to proceed.
<!-- MANUAL ADDITIONS END -->
