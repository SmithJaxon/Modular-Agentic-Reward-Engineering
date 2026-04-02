# Modular Agentic Reward Engineering

Research prototype for tool-based reward-function design in IsaacGym.

## Current Scope

- Phase 1 scaffold
- Shared project layout
- Environment-variable configuration
- Reproducible run manifests
- Placeholder experiment runner

## Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `PYTHON_VERSION_TARGET`
- `ISAACGYM_HOME`

See `.env.example` for a starter local environment file. Copy it to `.env` and fill in your values when ready.

## Layout

- `configs/`
- `runs/`
- `logs/`
- `artifacts/`
- `src/mare/`

## Phase 1 CLI

Example config:

- `configs/example_experiment.yaml`

Commands:

- `mare manifest --config configs/example_experiment.yaml`
- `mare dry-run --config configs/example_experiment.yaml`
- `mare list-presets`
- `mare preset --name cartpole`
- `mare script --config configs/example_experiment.yaml`
- `mare reward-validate --path reward_candidates/cartpole_reward.py`
- `mare reward-load --path reward_candidates/cartpole_reward.py`
- `mare recommend-reward-patch --path reward_candidates/cartpole_reward.py`
- `mare recommend-reward-patch --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare reward-robustness --path reward_candidates/cartpole_reward.py`
- `mare reward-robustness --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare run-summary --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare sweep-plan --config configs/example_experiment.yaml --reward-candidate reward_candidates/cartpole_reward.py`
- `mare sweep-plan --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare compare-report --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare review-brief --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare phase5-status --trace runs/cartpole_baseline_dry_run/orchestration.json`
- `mare benchmark-compare --baseline-run-dir runs/cartpole_baseline_dry_run_base --candidate-run-dir runs/cartpole_baseline_dry_run_seed_shift --metric evaluation_score`
- `mare benchmark-brief --baseline-run-dir runs/cartpole_baseline_dry_run_base --candidate-run-dir runs/cartpole_baseline_dry_run_seed_shift --metric evaluation_score`
- `mare benchmark-aggregate --run-dir runs/cartpole_baseline_dry_run --run-dir runs/cartpole_baseline_dry_run_base --metric evaluation_score`
- `mare benchmark-aggregate-brief --run-dir runs/cartpole_baseline_dry_run --run-dir runs/cartpole_baseline_dry_run_base --metric evaluation_score`
- `mare orchestrate --config configs/example_experiment.yaml --reward-candidate reward_candidates/cartpole_reward.py`

Reward candidates are Python modules with a required `compute_reward(observation, action, info)` function.
The `orchestrate` command currently uses a local heuristic policy and writes `orchestration.json` as the loop trace.
