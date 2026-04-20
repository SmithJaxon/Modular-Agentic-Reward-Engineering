#!/bin/bash
#SBATCH --job-name=rl-humanoid-human
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=notchpeak-gpu
#SBATCH --qos=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
SPEC_FILE="tools/fixtures/experiments/agent_isaacgym_humanoid_human_baseline.yaml"

utc_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

elapsed_seconds() {
  local start_epoch="$1"
  local end_epoch
  end_epoch="$(date +%s)"
  echo "$((end_epoch - start_epoch))"
}

run_and_time() {
  local label="$1"
  shift
  local start_epoch
  start_epoch="$(date +%s)"
  echo "[$label] start_utc=$(utc_now)"
  "$@"
  local status=$?
  echo "[$label] end_utc=$(utc_now) elapsed_s=$(elapsed_seconds "$start_epoch") status=$status"
  return $status
}

module --force purge
module load chpc/1.0
module load python/3.12.4

cd "$PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.venv-controller" ]; then
  echo "ERROR: controller venv not found at $PROJECT_ROOT/.venv-controller" >&2
  echo "Hint: submit with --export=ALL,PROJECT_ROOT=/absolute/repo/path" >&2
  exit 1
fi
if [ ! -f "$PROJECT_ROOT/$SPEC_FILE" ]; then
  echo "ERROR: spec file not found: $PROJECT_ROOT/$SPEC_FILE" >&2
  exit 1
fi
source .venv-controller/bin/activate

export PYTHONUNBUFFERED=1
export REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS="${REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS:-1800}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

if [ ! -f "$PROJECT_ROOT/tools/slurm/chpc_isaac_env.sh" ]; then
  echo "ERROR: Isaac env helper not found at $PROJECT_ROOT/tools/slurm/chpc_isaac_env.sh" >&2
  exit 1
fi
source "$PROJECT_ROOT/tools/slurm/chpc_isaac_env.sh"

echo "[diag] wrapper_start_utc=$(utc_now)"
echo "[diag] project_root=$PROJECT_ROOT"
echo "[diag] spec_file=$SPEC_FILE"
echo "[diag] slurm_job_id=${SLURM_JOB_ID:-unknown}"
echo "[diag] hostname=$(hostname)"
echo "[diag] pwd=$(pwd)"
echo "[diag] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[diag] isaac_run_timeout_seconds=$REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS"
echo "[diag] controller_python=$(command -v python)"
echo "[diag] worker_venv=${REWARDLAB_ISAAC_WORKER_VENV:-<unset>}"
echo "[diag] worker_command=${REWARDLAB_ISAAC_WORKER_COMMAND:-<unset>}"

echo "[diag] nvidia_smi_list"
nvidia-smi -L || true

echo "[diag] nvidia_smi_query"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

echo "[diag] controller_torch_probe"
python - <<'PY'
import json

payload = {"python_executable": __import__("sys").executable}
try:
    import torch
    payload["torch_importable"] = True
    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    payload["torch_importable"] = False
    payload["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
PY

echo "[diag] worker_torch_probe"
"$REWARDLAB_ISAAC_WORKER_VENV/bin/python" - <<'PY'
import json
import sys

payload = {"python_executable": sys.executable}
try:
    import torch
    payload["torch_importable"] = True
    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        payload["cuda_device_name_0"] = torch.cuda.get_device_name(0)
except Exception as exc:
    payload["torch_importable"] = False
    payload["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
PY

RUN_JSON="$PROJECT_ROOT/.rewardlab/chpc_full_diag_${SLURM_JOB_ID:-manual}.run.json"
rm -f "$RUN_JSON"

run_and_time preflight rewardlab experiment runtime-check --backend isaacgym --file "$SPEC_FILE" --json

run_start_epoch="$(date +%s)"
echo "[run] start_utc=$(utc_now) spec=$SPEC_FILE"
rewardlab experiment run --file "$SPEC_FILE" --json | tee "$RUN_JSON"
run_status=${PIPESTATUS[0]}
echo "[run] end_utc=$(utc_now) elapsed_s=$(elapsed_seconds "$run_start_epoch") status=$run_status"
if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi

if [ -f "$RUN_JSON" ]; then
  echo "[diag] run_json=$RUN_JSON"
  python - "$RUN_JSON" <<'PY'
import json
import pathlib
import sys

run_json_path = pathlib.Path(sys.argv[1])
payload = json.loads(run_json_path.read_text(encoding="utf-8"))
print(json.dumps(payload, sort_keys=True))
PY

  mapfile -t run_meta < <(python - "$RUN_JSON" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("experiment_id", ""))
print(payload.get("report_path", ""))
PY
  )
  experiment_id="${run_meta[0]:-}"
  report_path="${run_meta[1]:-}"

  if [ -n "$experiment_id" ] && [ -f "$PROJECT_ROOT/.rewardlab/events/events.jsonl" ]; then
    echo "[diag] filtered_events session_id=$experiment_id"
    python - "$PROJECT_ROOT/.rewardlab/events/events.jsonl" "$experiment_id" <<'PY'
import json
import pathlib
import sys

events_path = pathlib.Path(sys.argv[1])
session_id = sys.argv[2]
matches = []
for line in events_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("session_id") == session_id:
        matches.append(record)
for record in matches[-40:]:
    print(json.dumps(record, sort_keys=True))
PY
  fi

  if [ -n "$report_path" ] && [ -f "$PROJECT_ROOT/$report_path" ]; then
    echo "[diag] report_summary path=$report_path"
    python - "$PROJECT_ROOT/$report_path" <<'PY'
import json
import pathlib
import sys
from datetime import datetime

def parse(ts):
    if not isinstance(ts, str) or not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
experiment = report.get("experiment", {})
summary = report.get("summary", {})
started_at = parse(experiment.get("started_at"))
ended_at = parse(experiment.get("ended_at"))
elapsed_seconds = None
if started_at and ended_at:
    elapsed_seconds = (ended_at - started_at).total_seconds()
payload = {
    "experiment_id": experiment.get("experiment_id"),
    "status": experiment.get("status"),
    "stop_reason": experiment.get("stop_reason"),
    "elapsed_seconds": elapsed_seconds,
    "failed_actions": experiment.get("metadata", {}).get("failed_actions"),
    "consumed_train_timesteps": experiment.get("budget_ledger", {}).get("consumed_train_timesteps"),
    "run_count": summary.get("run_count"),
    "decision_count": summary.get("decision_count"),
    "best_score": summary.get("best_score"),
}
print(json.dumps(payload, sort_keys=True))
PY
  fi
fi

echo "[diag] wrapper_end_utc=$(utc_now)"
