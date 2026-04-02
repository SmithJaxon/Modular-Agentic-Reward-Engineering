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
- `mare orchestrate --config configs/example_experiment.yaml --reward-candidate reward_candidates/cartpole_reward.py`

Reward candidates are Python modules with a required `compute_reward(observation, action, info)` function.
The `orchestrate` command currently uses a local heuristic policy and writes `orchestration.json` as the loop trace.
