#!/bin/bash

resolve_rewardlab_job_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PROJECT_ROOT/$1" ;;
  esac
}

setup_rewardlab_job_runtime() {
  if [ -z "${PROJECT_ROOT:-}" ]; then
    echo "ERROR: PROJECT_ROOT must be set before calling setup_rewardlab_job_runtime" >&2
    return 1
  fi
  if [ -z "${SPEC_FILE:-}" ]; then
    echo "ERROR: SPEC_FILE must be set before calling setup_rewardlab_job_runtime" >&2
    return 1
  fi

  REWARDLAB_JOB_KEY="${SLURM_JOB_ID:-manual}"
  REWARDLAB_JOB_RUNTIME_ROOT_REL=".rewardlab_jobs/${REWARDLAB_JOB_KEY}"
  REWARDLAB_JOB_RUNTIME_ROOT="$(resolve_rewardlab_job_path "$REWARDLAB_JOB_RUNTIME_ROOT_REL")"
  REWARDLAB_JOB_SOURCE_SPEC_FILE="$(resolve_rewardlab_job_path "$SPEC_FILE")"
  REWARDLAB_JOB_SPEC_FILE_REL="${REWARDLAB_JOB_RUNTIME_ROOT_REL}/$(basename "$SPEC_FILE")"
  REWARDLAB_JOB_SPEC_FILE="$(resolve_rewardlab_job_path "$REWARDLAB_JOB_SPEC_FILE_REL")"
  REWARDLAB_JOB_EVENTS_JSON="${REWARDLAB_JOB_RUNTIME_ROOT}/events/events.jsonl"

  mkdir -p \
    "$REWARDLAB_JOB_RUNTIME_ROOT" \
    "$REWARDLAB_JOB_RUNTIME_ROOT/events" \
    "$REWARDLAB_JOB_RUNTIME_ROOT/checkpoints" \
    "$REWARDLAB_JOB_RUNTIME_ROOT/reports"

  export REWARDLAB_DATA_DIR="$REWARDLAB_JOB_RUNTIME_ROOT"
  export REWARDLAB_DB_PATH="$REWARDLAB_JOB_RUNTIME_ROOT/metadata.sqlite3"
  export REWARDLAB_EVENT_LOG_DIR="$REWARDLAB_JOB_RUNTIME_ROOT/events"
  export REWARDLAB_CHECKPOINT_DIR="$REWARDLAB_JOB_RUNTIME_ROOT/checkpoints"
  export REWARDLAB_REPORT_DIR="$REWARDLAB_JOB_RUNTIME_ROOT/reports"

  python - "$REWARDLAB_JOB_SOURCE_SPEC_FILE" "$REWARDLAB_JOB_SPEC_FILE" "$REWARDLAB_JOB_RUNTIME_ROOT_REL" <<'PY'
import pathlib
import sys

import yaml

source_path = pathlib.Path(sys.argv[1])
target_path = pathlib.Path(sys.argv[2])
runtime_dir = sys.argv[3]

payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("experiment spec must decode to a mapping")

outputs = payload.get("outputs")
if outputs is None:
    outputs = {}
elif not isinstance(outputs, dict):
    raise SystemExit("spec.outputs must decode to a mapping")
else:
    outputs = dict(outputs)

outputs["runtime_dir"] = runtime_dir
payload["outputs"] = outputs

target_path.parent.mkdir(parents=True, exist_ok=True)
target_path.write_text(
    yaml.safe_dump(payload, sort_keys=False),
    encoding="utf-8",
)
PY
}
