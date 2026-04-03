# Data Model: Real Experiment Readiness

## Reward Program

Purpose: Represents the executable reward logic derived from a stored candidate.

Fields:
- `candidate_id` (string, required)
- `entrypoint_name` (string, required, default `reward`)
- `source_text` (string, required)
- `signature_version` (string, required)
- `validation_status` (enum: valid, invalid)
- `validation_error` (string, optional)

Rules:
- The entrypoint MUST be validated before a real backend run starts.
- Invalid reward programs MUST produce actionable errors before backend execution.

## Actual Experiment Run

Purpose: Extends the existing experiment-run concept with explicit real-run
metadata and artifact linkage.

Fields:
- `run_id` (string, required)
- `candidate_id` (string, required)
- `backend` (enum: gymnasium, isaacgym)
- `environment_id` (string, required)
- `run_type` (enum: performance, reflection, robustness)
- `execution_mode` (enum: offline_test, actual_backend)
- `status` (enum: queued, running, paused, completed, failed)
- `metrics` (object, required for terminal successful runs)
- `artifact_refs` (list[string], default empty)
- `failure_reason` (string, optional)
- `started_at` (datetime, optional)
- `ended_at` (datetime, optional)

Rules:
- `execution_mode=actual_backend` MUST only be used when a real backend was instantiated.
- Successful actual runs MUST include non-empty metrics and at least one artifact
  manifest reference.

## Run Artifact Bundle

Purpose: Groups files created by a real experiment run.

Fields:
- `run_id` (string, required)
- `manifest_path` (string, required)
- `metrics_path` (string, required)
- `event_trace_path` (string, optional)
- `frame_dir` (string, optional)
- `video_path` (string, optional)

Rules:
- Every actual run MUST produce a manifest and metrics file.
- Optional media fields may be absent when rendering is unavailable.

## Backend Runtime Status

Purpose: Represents whether a backend can execute a real run in the current
approved worktree environment.

Fields:
- `backend` (enum: gymnasium, isaacgym)
- `ready` (boolean, required)
- `status_reason` (string, required)
- `missing_prerequisites` (list[string], default empty)
- `detected_version` (string, optional)

Rules:
- `ready=false` MUST include at least one actionable reason or prerequisite.
- Isaac runtime status MUST never claim readiness based solely on fake or test factories.

## Stored Robustness Assessment

Purpose: Persists robustness outcomes from real probe runs for later reporting
and candidate selection.

Fields:
- `assessment_id` (string, required)
- `candidate_id` (string, required)
- `backend` (enum: gymnasium, isaacgym)
- `primary_run_id` (string, required)
- `probe_run_ids` (list[string], required)
- `risk_level` (enum: low, medium, high)
- `risk_notes` (string, required)

Rules:
- Probe run ids MUST reference completed experiment runs.
- The assessment MUST be stored separately from transient orchestration state so
  reports and reviews can be reproduced later.
