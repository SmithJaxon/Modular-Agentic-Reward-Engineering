# Quickstart: Real Experiment Readiness

This tranche is now Gymnasium-only. Isaac runtime work is intentionally removed
from the active execution path because the current machine does not support it.

## Guardrails

- Work only in `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- Keep installs inside `.venv`
- Do not install anything without user approval
- Keep all temp files and artifacts inside the worktree

## Current Ready Path

The repo can already run a real Gymnasium smoke on `CartPole-v1`:

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

Important distinction:

- `REWARDLAB_EXECUTION_MODE=actual_backend` enables real Gymnasium execution
- `REWARDLAB_REWARD_DESIGN_MODE=openai` enables model-backed reward iteration

Without the second flag, `session step` still revises rewards with the
deterministic local designer, even though the backend execution is real.

## Humanoid PPO Target

The checked-in Humanoid fixtures are:

- `tools/fixtures/objectives/humanoid_run.txt`
- `tools/fixtures/rewards/humanoid_baseline.py`
- `tools/fixtures/experiments/gymnasium_humanoid.json`

The Humanoid path uses the PPO evaluation protocol in
`src/rewardlab/experiments/gymnasium_runner.py`:

- 5 PPO training runs
- 10 evaluation checkpoints per run
- final score = average of the per-run best checkpoint mean `x_velocity`

## Approval-Gated PPO Install

Humanoid PPO execution requires `stable-baselines3` in `.venv`.
If it is missing, ask the user before running a command like:

```powershell
.\.venv\Scripts\python -m pip install -e .[ppo]
```

After approval, record the exact command and detected versions in
`verification-report.md`.

## Humanoid Run Command

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

If `stable_baselines3` is missing, `session step` should fail with a clear
prerequisite error instead of falling back to the single-rollout heuristic.

## OpenAI Reward Iteration (Session Pipeline)

To run model-backed reward revision instead of the deterministic local revision
path, populate `.env` after user approval and set:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
$env:REWARDLAB_REWARD_DESIGN_MODE = "openai"
$env:REWARDLAB_REWARD_DESIGN_MODEL = "gpt-5-mini"
$env:REWARDLAB_REWARD_DESIGN_REASONING_EFFORT = "medium"
$env:REWARDLAB_REWARD_DESIGN_MAX_TOKENS = "2000"
```

The OpenAI-backed designer:

- uses the current reward candidate as the starting point
- includes the latest reflection summary and the latest run metrics in the prompt
- requires the returned reward code to stay executable and to use only supported
  callable parameters
- pauses the session with an explicit design error if the model returns invalid
  code or credentials are unavailable

Note:

- This is still the current `session step` pipeline.
- Planned autonomous controller/tool-calling mode is tracked in
  `specs/004-agent-tool-calling-architecture/`.

## Validation Commands

Offline and smoke validation:

```powershell
.\tools\quality\run_full_validation.ps1
.\tools\quality\run_real_backend_smokes.ps1
```
