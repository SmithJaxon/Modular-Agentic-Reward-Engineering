#!/bin/bash
#SBATCH --job-name=rl-humanoid-human
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SPEC_FILE="tools/fixtures/experiments/agent_isaacgym_humanoid_human_baseline.yaml"

module purge
module load python/3.12

cd "$PROJECT_ROOT"
source .venv-controller/bin/activate

export REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS="${REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS:-900}"
if [ -z "${REWARDLAB_ISAAC_WORKER_COMMAND:-}" ]; then
  export REWARDLAB_ISAAC_WORKER_COMMAND="$PROJECT_ROOT/.venv-isaac/bin/python $PROJECT_ROOT/tools/scripts/isaac_worker_py38.py"
fi

echo "[preflight] runtime-check"
rewardlab experiment runtime-check --backend isaacgym --file "$SPEC_FILE" --json

echo "[run] ${SPEC_FILE}"
rewardlab experiment run --file "$SPEC_FILE" --json

