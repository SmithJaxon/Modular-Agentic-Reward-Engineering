# Contract: Orchestrator CLI

## Overview

The CLI is the public interface for session orchestration and verification. All
commands MUST return machine-readable JSON when `--json` is supplied.

## Commands

### `rewardlab session start`

Starts a new optimization session.

Required arguments:
- `--objective-file <path>`
- `--baseline-reward-file <path>`
- `--environment-id <id>`
- `--environment-backend <gymnasium|isaacgym>`
- `--no-improve-limit <int>`
- `--max-iterations <int>`
- `--feedback-gate <none|one_required|both_required>`

Optional arguments:
- `--json`
- `--session-id <id>` (client-provided idempotent id)

Success response fields:
- `session_id`
- `status`
- `created_at`

### `rewardlab session step`

Executes one full iteration (evaluate, reflect, revise).

Required arguments:
- `--session-id <id>`

Optional arguments:
- `--json`

Success response fields:
- `session_id`
- `iteration_index`
- `candidate_id`
- `status`
- `best_candidate_id`

### `rewardlab session pause`

Pauses an active session explicitly.

Required arguments:
- `--session-id <id>`

### `rewardlab session resume`

Resumes a paused session.

Required arguments:
- `--session-id <id>`

### `rewardlab session stop`

Interrupts a running session and returns best-known candidate.

Required arguments:
- `--session-id <id>`

Success response fields:
- `session_id`
- `stop_reason`
- `best_candidate_id`
- `report_path`

### `rewardlab feedback submit-human`

Attaches human feedback for a candidate.

Required arguments:
- `--session-id <id>`
- `--candidate-id <id>`
- `--comment <text>`

Optional arguments:
- `--score <float>`
- `--artifact-ref <path-or-uri>`

### `rewardlab feedback request-peer`

Requests peer review from an isolated reviewer context.

Required arguments:
- `--session-id <id>`
- `--candidate-id <id>`

Success response fields:
- `feedback_id`
- `source_type`
- `comment`

### `rewardlab session report`

Exports final or intermediate session report.

Required arguments:
- `--session-id <id>`

Optional arguments:
- `--format <json|markdown>`
- `--output <path>`

## Error Contract

Errors MUST include:
- `error_code` (stable string)
- `message` (human-readable)
- `retryable` (boolean)
- `context` (optional object)
