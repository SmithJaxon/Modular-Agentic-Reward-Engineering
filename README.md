# RewardLab

RewardLab is a Python CLI for reward-function iteration and autonomous experiment control.

It supports:

- `rewardlab experiment ...` for autonomous experiment runs (recommended)
- `rewardlab session ...` for the legacy session pipeline

## Setup (Windows and Linux/macOS)

Use one virtual environment per repo checkout. Recommended folder name: `.venv`.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Linux/macOS (bash/zsh)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional dev tooling:

```bash
python -m pip install -r requirements-dev.txt
```

Important:

- Do not mix commands across multiple venvs for one run.
- If you accidentally created multiple envs, pick one and reinstall there.

## Dependency Profiles

- `requirements.txt`: base runtime
- `requirements-dev.txt`: base + lint/test/typecheck tooling
- `requirements-runtime-gymnasium.txt`: base + Gymnasium MuJoCo runtime
- `requirements-runtime-humanoid.txt`: Gymnasium runtime + Torch + SB3
- `requirements-all.txt`: dev + Humanoid runtime bundle

For Humanoid PPO runtime:

```bash
python -m pip install -r requirements-runtime-humanoid.txt
```

## CLI Quick Check

After venv activation:

```bash
rewardlab --help
rewardlab experiment --help
```

## Autonomous Experiments (Beta)

Validate a spec:

```bash
rewardlab experiment validate --file tools/fixtures/experiments/agent_humanoid_balanced.yaml --json
```

Run an experiment:

```bash
rewardlab experiment run --file tools/fixtures/experiments/agent_humanoid_balanced.yaml --json
```

Status and trace:

```bash
rewardlab experiment status --experiment-id <EXPERIMENT_ID> --json
rewardlab experiment trace --experiment-id <EXPERIMENT_ID> --json
```

Benchmark run:

```bash
rewardlab experiment benchmark-run --file tools/fixtures/experiments/agent_humanoid_balanced.yaml --seed 1 --seed 2 --seed 3 --json
```

### Large Eureka-style fixture

```bash
rewardlab experiment run --file tools/fixtures/experiments/agent_humanoid_eureka_5x16_large.yaml --json
```

### Quick affordable fixture

```bash
rewardlab experiment run --file tools/fixtures/experiments/agent_humanoid_quick_affordable.yaml --json
```

## Agent Loop Controls

Use these fields in spec `agent_loop` and `governance.stopping` to control when the agent keeps searching vs stops:

- `agent_loop.samples_per_iteration`: reward proposals to create for each iteration target.
- `agent_loop.encourage_run_all_after_each_experiment`: adds guidance to run pending candidates before proposing more.
- `agent_loop.enforce_progress_before_stop` (default `true`): blocks controller/tool `stop` actions until iteration/sample targets are met, unless a hard stop policy is hit.
- `governance.stopping.max_iterations`: iteration target/cap for the run.

Hard stop reasons are still respected even with progress gating:

- API token/USD budget exhausted
- compute experiment/timestep/reward-generation budget exhausted
- wall-clock budget exhausted
- failed-action threshold reached

### Why `estimate_cost_and_risk` appears

`estimate_cost_and_risk` is an analyzer action. It does not stop the run by itself. It is used to summarize budget utilization and risk before deciding the next action.

## Legacy Session Pipeline

Start:

```bash
rewardlab session start --objective-file tools/fixtures/objectives/cartpole.txt --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py --environment-id CartPole-v1 --environment-backend gymnasium --no-improve-limit 3 --max-iterations 5 --feedback-gate none --json
```

Step/stop/report:

```bash
rewardlab session step --session-id <SESSION_ID> --json
rewardlab session stop --session-id <SESSION_ID> --json
rewardlab session report --session-id <SESSION_ID> --json
```

## Output, Logs, Artifacts

Default runtime root is `.rewardlab/`:

- `.rewardlab/metadata.sqlite3`
- `.rewardlab/events/events.jsonl`
- `.rewardlab/runs/`
- `.rewardlab/reports/agent_experiments/`
- `.rewardlab/reports/agent_benchmarks/`

## Common Errors

### `No such command 'agent'`

Use `rewardlab experiment run --file <SPEC_PATH> --json`, not `rewardlab agent run ...`.

### `PyYAML is required to load YAML experiment specs`

Install base requirements in the active venv:

```bash
python -m pip install -r requirements.txt
```

### MuJoCo install/build errors

If `pip` tries to build `mujoco` from source and fails, use Python 3.12 or 3.13 in your venv, then reinstall runtime dependencies.

## Environment Variables

- `OPENAI_API_KEY`
- `REWARDLAB_EXECUTION_MODE` (`offline_test` or `actual_backend`)
- `REWARDLAB_REWARD_DESIGN_MODE` (`deterministic` or `openai`)
- `REWARDLAB_PPO_TOTAL_TIMESTEPS`
- `REWARDLAB_PPO_EVAL_RUNS`
- `REWARDLAB_PPO_CHECKPOINT_COUNT`
- `REWARDLAB_PPO_EVAL_EPISODES`

## Dev Checks

```bash
cd src
pytest
ruff check .
```
