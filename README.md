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

### Eureka-style comparison metrics

Compute Human Normalized Score (HNS) and optional reward-hacking metrics:

```bash
rewardlab experiment eureka-metrics \
  --method-report .rewardlab/reports/agent_experiments/<METHOD_REPORT>.report.json \
  --human-score <HUMAN_SCORE> \
  --sparse-score <SPARSE_SCORE> \
  --probe-score <PROBE_SCORE_1> \
  --probe-score <PROBE_SCORE_2> \
  --json
```

You can also resolve scores directly from report files:

```bash
rewardlab experiment eureka-metrics \
  --method-report <METHOD_REPORT_PATH> \
  --human-report <HUMAN_BASELINE_REPORT_PATH> \
  --sparse-report <SPARSE_BASELINE_REPORT_PATH> \
  --probe-report <ROBUSTNESS_PROBE_REPORT_1> \
  --probe-report <ROBUSTNESS_PROBE_REPORT_2> \
  --json
```

Notes:

- HNS follows Eureka's method-vs-sparse normalization with absolute human-sparse scaling.
- Output includes both raw HNS and clipped HNS (`--clip-min`, `--clip-max`, defaults `0.0..3.0`).
- Reward-hacking output includes a Perils-inspired scalar:
  `perils_relative_reward_function_performance` (probe-context score relative to nominal-context score).

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
- `initialization.mode`: first-candidate seed mode (`human` or `default`).
  - `human`: candidate-000 loads from `baseline_reward.path`.
  - `default`: candidate-000 is generated through the reward-designer bootstrap path (Eureka-like
    zero-shot seed), with deterministic/template fallback if model output is unavailable.
- `initialization.default_seed_candidate_count` (default `1`): number of i.i.d bootstrap seed
  candidates created at experiment start when `initialization.mode=default`.

Hard stop reasons are still respected even with progress gating:

- API token/USD budget exhausted
- compute experiment/timestep/reward-generation budget exhausted
- wall-clock budget exhausted
- failed-action threshold reached

PPO execution tuning (Humanoid):

- `execution.ppo.n_envs`: number of parallel training environments per candidate (`1` = single env, `>1` = subprocess vectorized envs).
- `execution.ppo.device`: SB3/PyTorch device string (`auto`, `cpu`, `cuda`, `cuda:0`, etc.).
- `budgets.compute.max_parallel_experiments`: max number of pending candidate evaluations dispatched in parallel per `run_experiment` action.

Comparison metrics in final reports:

- `execution.comparison.enabled`: enable auto-computation of HNS and probe-based hacking metrics at report time.
- `execution.comparison.human_reward_path`: reward program used as the human reference.
- `execution.comparison.sparse_reward_path`: reward program used as the sparse reference.
- `execution.comparison.num_eval_runs` / `seed_start`: evaluation schedule for method/human/sparse score blocks.
- `execution.comparison.probe_run_count` / `probe_seed_start`: probe schedule for hacking metrics.
- `comparison_metrics.reward_hacking.perils_relative_reward_function_performance`: single scalar
  for cross-method reward-hacking comparison (higher is better; values near `1.0` indicate stronger
  retention under probe contexts).
  - Implemented as `probe_mean_score / abs(method_score)` and reported alongside
    `perils_hacking_severity = 1 - clip(relative_performance, 0, 1)`.

Recommended scheduling on one RTX 3090 and ~18 CPU cores:

- Start with `execution.ppo.n_envs: 6` to `8`.
- Start with `budgets.compute.max_parallel_experiments: 2`.
- Increase only one dimension at a time (`n_envs` first, then `max_parallel_experiments`) while watching CPU saturation and GPU memory.

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

Agent-experiment report filenames now use:

- `<spec-file-stem>--<experiment-id>.report.json`

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
- `REWARDLAB_PPO_N_ENVS`
- `REWARDLAB_PPO_DEVICE`

## Dev Checks

```bash
cd src
pytest
ruff check .
```
