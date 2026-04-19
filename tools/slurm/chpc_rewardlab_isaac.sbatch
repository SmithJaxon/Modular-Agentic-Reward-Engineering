#!/bin/bash
#SBATCH --job-name=rewardlab-isaac
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
SPEC_FILE="${SPEC_FILE:-tools/fixtures/experiments/agent_isaacgym_chpc_profile.yaml}"

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

export REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS="${REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS:-900}"
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

# Optionally point at a dedicated Isaac worker runtime.
# export REWARDLAB_ISAAC_WORKER_COMMAND="python3.8 tools/scripts/isaac_worker_py38.py"
# export REWARDLAB_ISAAC_CFG_DIR="/path/to/IsaacGymEnvs/isaacgymenvs/cfg"

echo "[preflight] runtime-check"
rewardlab experiment runtime-check --backend isaacgym --json

echo "[run] ${SPEC_FILE}"
rewardlab experiment run --file "$SPEC_FILE" --json


