#!/bin/bash
#SBATCH --job-name=rl-humanoid-otherresults
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
SPEC_FILE="tools/fixtures/experiments/agent_isaacgym_humanoid_otherresults_eval.yaml"

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

if [ ! -f "$PROJECT_ROOT/tools/slurm/chpc_rewardlab_job_runtime.sh" ]; then
  echo "ERROR: runtime isolation helper not found at $PROJECT_ROOT/tools/slurm/chpc_rewardlab_job_runtime.sh" >&2
  exit 1
fi
source "$PROJECT_ROOT/tools/slurm/chpc_rewardlab_job_runtime.sh"
setup_rewardlab_job_runtime

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

echo "[preflight] runtime-check"
echo "[diag] source_spec_file=$SPEC_FILE"
echo "[diag] effective_spec_file=$REWARDLAB_JOB_SPEC_FILE_REL"
echo "[diag] job_runtime_root=$REWARDLAB_JOB_RUNTIME_ROOT"
echo "[diag] rewardlab_db_path=$REWARDLAB_DB_PATH"
rewardlab experiment runtime-check --backend isaacgym --file "$REWARDLAB_JOB_SPEC_FILE" --json

echo "[run] ${REWARDLAB_JOB_SPEC_FILE_REL}"
rewardlab experiment run --file "$REWARDLAB_JOB_SPEC_FILE" --json

