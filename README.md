# RewardLab

RewardLab is a Python CLI for reward-function iteration and autonomous experiment control.

It currently supports:

- `rewardlab experiment ...`: autonomous experiment workflow (recommended for new runs)
- `rewardlab session ...`: legacy session pipeline

## Setup (Use One Virtual Environment)

Use exactly one worktree-local venv for this repo. Recommended name: `.venv`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional dev tooling:

```powershell
python -m pip install -r requirements-dev.txt
```

Important:

- Do not mix commands across `.venv` and `.venv-mujoco` in one run.
- If you already created multiple envs, pick one and use it consistently.

## Install Optional Runtime Extras

For Humanoid PPO and real Gymnasium execution:

```powershell
python -m pip install -r requirements-runtime-humanoid.txt
```

This installs Gymnasium MuJoCo support, Torch, and Stable-Baselines3 in the active venv.

Install profiles:

- `requirements.txt`: base CLI/runtime
- `requirements-dev.txt`: base + lint/test/typecheck toolchain
- `requirements-runtime-gymnasium.txt`: base + Gymnasium MuJoCo runtime
- `requirements-runtime-humanoid.txt`: Gymnasium runtime + PPO stack (Torch + SB3)
- `requirements-all.txt`: dev + Humanoid runtime convenience install

## CLI Quick Check

```powershell
.\.venv\Scripts\rewardlab.exe --help
.\.venv\Scripts\rewardlab.exe experiment --help
```

## Run Autonomous Experiments (Beta)

Validate a spec:

```powershell
.\.venv\Scripts\rewardlab.exe experiment validate `
  --file tools\fixtures\experiments\agent_humanoid_balanced.yaml `
  --json
```

Run an experiment:

```powershell
.\.venv\Scripts\rewardlab.exe experiment run `
  --file tools\fixtures\experiments\agent_humanoid_balanced.yaml `
  --json
```

Check status and trace:

```powershell
.\.venv\Scripts\rewardlab.exe experiment status --experiment-id <EXPERIMENT_ID> --json
.\.venv\Scripts\rewardlab.exe experiment trace --experiment-id <EXPERIMENT_ID> --json
```

Run benchmark seeds:

```powershell
.\.venv\Scripts\rewardlab.exe experiment benchmark-run `
  --file tools\fixtures\experiments\agent_humanoid_balanced.yaml `
  --seed 1 --seed 2 --seed 3 `
  --json
```

## Run the Large Eureka-Style Fixture

```powershell
.\.venv\Scripts\rewardlab.exe experiment run `
  --file tools\fixtures\experiments\agent_humanoid_eureka_5x16_large.yaml `
  --json
```

This fixture is configured for:

- 5 reward generations
- 16 samples per generation
- Humanoid PPO (`total_timesteps: 100000`, `eval_runs: 5`)
- final multi-seed evaluation (`execution.final_evaluation.num_eval_runs: 5`)

## Legacy Session Pipeline

Start a local CartPole session:

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools\fixtures\objectives\cartpole.txt `
  --baseline-reward-file tools\fixtures\rewards\cartpole_baseline.py `
  --environment-id CartPole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --json
```

Then:

```powershell
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session report --session-id <SESSION_ID> --json
```

## Output, Logs, and Artifacts

By default, runtime output is under `.rewardlab/`.

- `.\.rewardlab\metadata.sqlite3`: metadata store
- `.\.rewardlab\events\events.jsonl`: event log stream
- `.\.rewardlab\runs\`: per-run manifests/metrics
- `.\.rewardlab\reports\agent_experiments\`: autonomous experiment reports
- `.\.rewardlab\reports\agent_benchmarks\`: benchmark aggregate reports

## Common Errors

### `No such command 'agent'`

Use:

```powershell
rewardlab experiment run --file <SPEC_PATH> --json
```

Not:

```powershell
rewardlab agent run ...
```

### `PyYAML is required to load YAML experiment specs`

Install base requirements in the active venv:

```powershell
python -m pip install -r requirements.txt
```

Or run JSON specs instead of YAML.

### Two-venv confusion

Symptoms: command exists in one venv, dependencies in another.

Fix:

1. Activate only one venv for the entire run.
2. Reinstall requirements in that same venv.
3. Run all `rewardlab` commands from that same shell session.

## Environment Variables

- `OPENAI_API_KEY`: required for model-backed paths
- `REWARDLAB_EXECUTION_MODE`: `offline_test` or `actual_backend`
- `REWARDLAB_REWARD_DESIGN_MODE`: `deterministic` or `openai`

Humanoid PPO knobs:

- `REWARDLAB_PPO_TOTAL_TIMESTEPS`
- `REWARDLAB_PPO_EVAL_RUNS`
- `REWARDLAB_PPO_CHECKPOINT_COUNT`
- `REWARDLAB_PPO_EVAL_EPISODES`

## Dev Checks

```powershell
cd src
pytest
ruff check .
```
