# RewardLab

RewardLab is a local-first Python CLI for iterative reward-function design.
It provides explicit reward-designer modes, backend routing, robustness
assessment hooks, feedback gating, and interruption-safe report export.

RewardLab now includes a beta autonomous experiment path via
`rewardlab experiment ...` commands, alongside the legacy `session` pipeline.
The architecture plan is documented in
`specs/004-agent-tool-calling-architecture/plan.md`.

## Autonomous Experiment Workflow (Beta)

Validate and run from an experiment spec:

```powershell
.\.venv\Scripts\rewardlab.exe experiment validate `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --json
.\.venv\Scripts\rewardlab.exe experiment run `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --json
```

Read status and full decision trace:

```powershell
.\.venv\Scripts\rewardlab.exe experiment status `
  --experiment-id <EXPERIMENT_ID> `
  --json
.\.venv\Scripts\rewardlab.exe experiment trace `
  --experiment-id <EXPERIMENT_ID> `
  --json
```

Run a multi-seed benchmark with aggregate stats:

```powershell
.\.venv\Scripts\rewardlab.exe experiment benchmark-run `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --seed 1 `
  --seed 2 `
  --seed 3 `
  --json
```

Benchmark reports are written under
`.rewardlab/reports/agent_benchmarks/` and include score distributions,
improvement rates, confidence intervals, action-mix totals, stop-reason counts,
and resource-efficiency metrics.

If autonomous control pauses for human feedback (`awaiting_human_feedback`):

```powershell
.\.venv\Scripts\rewardlab.exe experiment submit-human-feedback `
  --experiment-id <EXPERIMENT_ID> `
  --candidate-id <CANDIDATE_ID> `
  --comment "Reasonable progress; continue exploring." `
  --request-id <REQUEST_ID> `
  --json
.\.venv\Scripts\rewardlab.exe experiment resume `
  --experiment-id <EXPERIMENT_ID> `
  --json
```

Reference experiment specs:

- `tools/fixtures/experiments/agent_humanoid_balanced.yaml`
- `tools/fixtures/experiments/agent_humanoid_high_budget.yaml`
- `tools/fixtures/experiments/agent_cartpole_lowcost.yaml`

## What It Does

- Starts and manages reward-optimization sessions from an objective file and
  baseline reward definition.
- Runs either a deterministic offline iteration loop or an explicit
  OpenAI-backed reward-design loop.
- Routes experiment execution through the active `gymnasium` backend.
- Records human feedback and peer feedback with session-level gating rules.
- Persists checkpoints, event logs, and JSON reports under a worktree-local
  `.rewardlab/` runtime directory.

## Local Setup

Create a worktree-local virtual environment and install the package in editable
mode:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
```

No machine-level installs, PATH changes, or global Python package changes are
required.

## Real Backend Prerequisites

Real backend execution is approval-gated and is not part of the default offline
setup. Keep every install inside this worktree's `.venv` and record the exact
approved command once it is run.

Common install shapes after approval:

```powershell
.\.venv\Scripts\python -m pip install -e .[dev,gymnasium,torch]
.\.venv\Scripts\python -m pip install -e .[dev,gymnasium,torch,ppo]
```

Default `pytest` runs skip the approval-gated real backend smokes. Opt in only
after the required runtime is present:

```powershell
.\.venv\Scripts\python -m pytest --run-real-gymnasium
```

To route `session step` through the real backend path instead of the offline
heuristic path, set:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
```

Use `offline_test` to force the deterministic validation mode.

## Reward Designer Modes

RewardLab now separates backend execution from reward generation:

- `REWARDLAB_EXECUTION_MODE=offline_test`:
  deterministic local scoring and deterministic local reward revision
- `REWARDLAB_EXECUTION_MODE=actual_backend` plus default reward-designer mode:
  real Gymnasium execution, but still deterministic local reward revision
- `REWARDLAB_EXECUTION_MODE=actual_backend` plus
  `REWARDLAB_REWARD_DESIGN_MODE=openai`:
  real Gymnasium execution and model-backed reward iteration

This matters for Humanoid experiments: the first real PPO session can cost
`$0` in tokens if the reward-designer mode stays deterministic. To run the
original agent-driven search loop, enable the OpenAI reward designer
explicitly.

## Offline Workflow

Most validation and CLI flows run without an API key. The peer-feedback path
uses a deterministic local fallback unless `OPENAI_API_KEY` is set, and the
reward-design loop stays deterministic unless
`REWARDLAB_REWARD_DESIGN_MODE=openai` is set.

Start a session with the checked-in CartPole fixtures:

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id CartPole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --json
```

Step the session, attach feedback, and export a report:

```powershell
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe feedback submit-human `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --comment "Stable behavior with acceptable oscillation." `
  --score 0.8 `
  --artifact-ref demo.md `
  --json
.\.venv\Scripts\rewardlab.exe feedback request-peer `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --json
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session report --session-id <SESSION_ID> --json
```

## Actual Gymnasium Workflow

After approval-gated Gymnasium dependencies are installed, enable the real
backend mode and run the same CLI flow against `CartPole-v1`:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id CartPole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 2 `
  --max-iterations 2 `
  --feedback-gate none `
  --json
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
```

In real backend mode, RewardLab writes per-run manifests and metrics beneath
`.rewardlab/runs/`, persists `ExperimentRun` records in SQLite, and includes run
ids plus artifact references in the exported report summaries.

If you do not also enable `REWARDLAB_REWARD_DESIGN_MODE=openai`, the iteration
loop still uses the deterministic local reward designer.

## Humanoid PPO Workflow

Gymnasium Humanoid uses a PPO-based evaluation path instead of the lightweight
single-rollout smoke path. The checked-in protocol measures candidate quality as
the average across 5 PPO runs of the best checkpoint mean `x_velocity` observed
over 10 evaluation checkpoints, matching the EUREKA paper's evaluation shape.

Humanoid PPO requires `stable-baselines3` in the active `.venv`.
If it is not present, install it with user approval.

Recommended command shape after approval:

```powershell
.\.venv\Scripts\python -m pip install -e .[ppo]
```

Humanoid fixture files are checked in at:

- `tools/fixtures/objectives/humanoid_run.txt`
- `tools/fixtures/rewards/humanoid_baseline.py`
- `tools/fixtures/experiments/gymnasium_humanoid.json`

Example run:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
$env:REWARDLAB_PPO_TOTAL_TIMESTEPS = "50000"
$env:REWARDLAB_PPO_EVAL_RUNS = "5"
$env:REWARDLAB_PPO_CHECKPOINT_COUNT = "10"
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/humanoid_run.txt `
  --baseline-reward-file tools/fixtures/rewards/humanoid_baseline.py `
  --environment-id Humanoid-v4 `
  --environment-backend gymnasium `
  --no-improve-limit 2 `
  --max-iterations 2 `
  --feedback-gate none `
  --json
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
```

If `stable_baselines3` is missing, the step fails with an explicit prerequisite
message rather than silently falling back to the rollout heuristic.

## OpenAI-Backed Reward Iteration

To run model-backed reward iteration inside the current `session` pipeline,
approve `.env` usage, then set the reward-designer mode explicitly before
`session step`:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
$env:REWARDLAB_REWARD_DESIGN_MODE = "openai"
$env:REWARDLAB_REWARD_DESIGN_MODEL = "gpt-5-mini"
$env:REWARDLAB_REWARD_DESIGN_REASONING_EFFORT = "medium"
$env:REWARDLAB_REWARD_DESIGN_MAX_TOKENS = "2000"
```

Optional tuning:

- `REWARDLAB_REWARD_DESIGN_MODEL`: override the reward-generation model
- `REWARDLAB_REWARD_DESIGN_REASONING_EFFORT`: one of `minimal`, `low`,
  `medium`, `high`
- `REWARDLAB_REWARD_DESIGN_MAX_TOKENS`: response budget for one reward revision

In this mode, each `session step` uses the latest candidate, the latest
reflection, and the latest run metrics to ask the model for the next executable
reward definition. If the model returns invalid code, unsupported callable
parameters, or no credentials are available, RewardLab pauses the session with
an explicit design error instead of silently fabricating another candidate.

Generated artifacts are written beneath `.rewardlab/`:

- `metadata.sqlite3`: session and namespaced metadata index
- `events/events.jsonl`: append-only lifecycle and feedback events
- `checkpoints/`: resumable session snapshots
- `reports/<session-id>.report.json`: exported session reports
- `reports/feedback_artifacts/`: human-review artifact bundles

## Optional API-Backed Peer Review

Copy `.env.example` to `.env` only when live peer-feedback calls are needed:

```powershell
Copy-Item .env.example .env
```

Populate `OPENAI_API_KEY` in `.env`. Live peer feedback uses the low-cost
`gpt-5-nano` model with a single short critique prompt. Model-backed reward
iteration uses the configured `REWARDLAB_REWARD_DESIGN_MODEL`. If no key is
present, RewardLab stays offline and uses the deterministic fallback path
instead. RewardLab auto-loads the nearest local `.env` file from the current
working directory upward, while process environment variables still take
precedence.

## Quality Gate

Run the full local validation suite with one command:

```powershell
.\tools\quality\run_full_validation.ps1
```

Run the opt-in real backend smokes with:

```powershell
.\tools\quality\run_real_backend_smokes.ps1
```

That runner executes:

- contract/schema validation
- file-header and docstring audit
- `ruff`
- `mypy`
- all unit, contract, integration, and end-to-end tests

The real-backend smoke wrapper now runs only the Gymnasium smoke.

Detailed operator steps and manual workflow examples are in
`specs/001-iterative-reward-design/quickstart.md` and
`specs/003-real-experiment-readiness/quickstart.md`.
