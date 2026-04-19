#!/bin/bash
#SBATCH --job-name=rewardlab-isaac
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SPEC_FILE="${SPEC_FILE:-tools/fixtures/experiments/agent_isaacgym_chpc_profile.yaml}"

module purge
module load python/3.12

cd "$PROJECT_ROOT"
source .venv-controller/bin/activate

export REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS="${REWARDLAB_ISAAC_RUN_TIMEOUT_SECONDS:-900}"
# Optionally point at a dedicated Isaac worker runtime.
# export REWARDLAB_ISAAC_WORKER_COMMAND="python3.8 -m rewardlab.experiments.isaacgym_worker"
# export REWARDLAB_ISAAC_CFG_DIR="/path/to/IsaacGymEnvs/isaacgymenvs/cfg"

echo "[preflight] runtime-check"
rewardlab experiment runtime-check --backend isaacgym --json

echo "[run] ${SPEC_FILE}"
rewardlab experiment run --file "$SPEC_FILE" --json
