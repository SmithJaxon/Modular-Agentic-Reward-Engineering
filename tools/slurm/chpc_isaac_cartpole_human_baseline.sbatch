#!/bin/bash
#SBATCH --job-name=rl-cartpole-human
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=notchpeak-gpu
#SBATCH --qos=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
SPEC_FILE="tools/fixtures/experiments/agent_isaacgym_cartpole_human_baseline.yaml"

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

if [ -z "${REWARDLAB_ISAAC_WORKER_COMMAND:-}" ]; then
  export REWARDLAB_ISAAC_WORKER_COMMAND="$PROJECT_ROOT/.venv-isaac/bin/python $PROJECT_ROOT/tools/scripts/isaac_worker_py38.py"
fi

echo "[preflight] runtime-check"
rewardlab experiment runtime-check --backend isaacgym --file "$SPEC_FILE" --json

echo "[run] ${SPEC_FILE}"
rewardlab experiment run --file "$SPEC_FILE" --json

