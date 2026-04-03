#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
while [ ! -f "${PROJECT_ROOT}/pyproject.toml" ] && [ "${PROJECT_ROOT}" != "/" ]; do
  PROJECT_ROOT="$(dirname "${PROJECT_ROOT}")"
done
if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
  echo "Unable to locate project root from ${SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi
python3 scripts/train_ppo.py --environment CartPole --task-name cartpole --algorithm PPO --seed 7 --train-steps 50000 --eval-episodes 10 --device auto --run-dir "${SCRIPT_DIR}" --reward-candidate reward_candidates/cartpole_reward.py --reward-entrypoint compute_reward
