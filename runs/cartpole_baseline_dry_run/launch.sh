#!/usr/bin/env bash
set -euo pipefail
cd "/Users/isabella/MARE/Modular-Agentic-Reward-Engineering"
if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi
python3 scripts/train_ppo.py --environment CartPole --task-name cartpole --algorithm PPO --seed 7 --train-steps 50000 --eval-episodes 10 --device cuda --run-dir /Users/isabella/MARE/Modular-Agentic-Reward-Engineering/runs/cartpole_baseline_dry_run
