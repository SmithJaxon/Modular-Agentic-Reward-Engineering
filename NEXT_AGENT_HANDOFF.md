# Next Agent Handoff

## Superseded Notice (2026-04-10)

This handoff reflects the finalized `003-real-experiment-readiness` tranche and
contains historical blockers that have since been resolved.

For active implementation direction, use:

- `specs/004-agent-tool-calling-architecture/plan.md`
- `specs/004-agent-tool-calling-architecture/quickstart.md`

## Workspace

- Active worktree:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- Stay inside this worktree only.
- Keep installs, temp files, artifacts, and reports inside this worktree.

## User Direction

- Isaac is out of the active runtime path because the current machine does not
  support it.
- The project should focus on Gymnasium only.
- The target is a full Gymnasium Humanoid experiment loop where:
  - PPO evaluation follows the EUREKA-style shape
  - the agent actually iterates on reward functions
  - the next reward candidate is generated from prior reward code, reflection,
    and run metrics

## Current State

### Runtime Scope

- Active backend: `gymnasium`
- Removed from the active path:
  - Isaac-specific runtime branches
  - Isaac-specific tests
  - Isaac-specific fixtures
  - Isaac-specific contract/tooling references

### Humanoid PPO Path

- `src/rewardlab/experiments/gymnasium_runner.py` implements Gymnasium
  Humanoid PPO evaluation for:
  - `Humanoid-v4`
  - `Humanoid-v5`
- Evaluation protocol:
  - 5 PPO runs
  - 10 checkpoints per run
  - final score = average of the per-run best checkpoint mean `x_velocity`
- Missing `stable_baselines3` now fails explicitly instead of silently falling
  back.

### Reward Iteration Path

- `src/rewardlab/orchestrator/reward_designer.py` now exists and is the key new
  integration point.
- Reward-designer modes:
  - `deterministic`
  - `openai`
- `src/rewardlab/orchestrator/session_service.py` now routes candidate
  generation through the configured reward designer.
- Actual-backend iteration now includes:
  - current reward candidate
  - latest reflection
  - latest completed run metrics
  - environment id/backend context
- Invalid model output, unsupported callable parameters, or missing OpenAI
  credentials now pause the session with an explicit design error.

## Important Distinction

This was the main confusion in the previous pass:

- `REWARDLAB_EXECUTION_MODE=actual_backend`
  means real Gymnasium execution
- `REWARDLAB_REWARD_DESIGN_MODE=openai`
  means model-backed reward iteration

If the second flag is not set, the system still uses the deterministic local
reward designer even though PPO execution is real. That is why the earlier
Humanoid PPO run consumed no tokens.

## What Changed In This Pass

### Earlier Gymnasium Pivot

- Removed Isaac-specific runtime code/tests/contracts/tooling.
- Kept real Gymnasium smoke and artifact/report plumbing working.
- Added Humanoid fixtures:
  - `tools/fixtures/objectives/humanoid_run.txt`
  - `tools/fixtures/rewards/humanoid_baseline.py`
  - `tools/fixtures/experiments/gymnasium_humanoid.json`

### Real Humanoid PPO Support

- Implemented actual Humanoid PPO evaluation in:
  - `src/rewardlab/experiments/gymnasium_runner.py`
- Installed approved PPO dependency in `.venv`:
  - `stable_baselines3==2.8.0`

### New Agent-Driven Reward Iteration

- Added:
  - `src/rewardlab/orchestrator/reward_designer.py`
  - `tests/unit/test_reward_designer.py`
- Updated:
  - `src/rewardlab/orchestrator/iteration_engine.py`
  - `src/rewardlab/orchestrator/session_service.py`
  - `tests/integration/test_gymnasium_real_experiment.py`
- Rewrote active docs to explain the new explicit reward-designer mode.

## Real Humanoid Baseline Run Already Completed

The worktree has already completed one real `Humanoid-v4` PPO session.

### Session

- Session id: `session-real-humanoid-ppo-20260406`
- Environment: `Humanoid-v4`
- Backend: `gymnasium`
- Execution mode: `actual_backend`

### Result

- `fitness_metric_name`: `mean_x_velocity`
- `fitness_metric_mean`: `0.482898`
- `per_run_best_mean_x_velocity`:
  - `0.577095`
  - `0.441847`
  - `0.515695`
  - `0.531256`
  - `0.348597`

### Persisted Evidence

- Manifest:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\runs\session-real-humanoid-ppo-20260406-run-001\manifest.json`
- Metrics:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\runs\session-real-humanoid-ppo-20260406-run-001\metrics.json`
- Report:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\reports\session-real-humanoid-ppo-20260406.report.json`

Important:

- This run used the real PPO backend.
- It predated the new OpenAI reward-designer path.
- It is therefore a real execution baseline, not a real token-consuming reward
  search baseline.

## Validation Completed

### Earlier Runtime Validation

- `ruff` on touched Gymnasium/runtime files: PASS
- targeted mypy on touched runtime files: PASS
- `tests/integration/test_gymnasium_real_experiment.py`: PASS
- real Gymnasium smoke wrapper: PASS

### Reward-Designer Validation

- `ruff` on:
  - `src/rewardlab/orchestrator/reward_designer.py`
  - `src/rewardlab/orchestrator/iteration_engine.py`
  - `src/rewardlab/orchestrator/session_service.py`
  - new/updated tests
  - Result: PASS
- targeted mypy on:
  - `src/rewardlab/orchestrator/reward_designer.py`
  - `src/rewardlab/orchestrator/iteration_engine.py`
  - `src/rewardlab/orchestrator/session_service.py`
  - Result: PASS
- focused pytest:
  - `tests/unit/test_reward_designer.py`
  - `tests/integration/test_gymnasium_real_experiment.py`
  - Result: `10 passed`

### Known Windows Validation Wrinkle

- Broader pytest slices can still fail during teardown with a Windows
  `basetemp` cleanup permission error.
- This is an environment/filesystem cleanup issue, not a reward-designer logic
  failure.
- The focused reward-designer tests passed.

## Key Code References

- Reward-designer config and mode selection:
  - `src/rewardlab/orchestrator/reward_designer.py`
- OpenAI-backed reward generation:
  - `src/rewardlab/orchestrator/reward_designer.py`
- Reward iteration plumbing:
  - `src/rewardlab/orchestrator/iteration_engine.py`
- Actual-backend integration and pause-on-design-failure behavior:
  - `src/rewardlab/orchestrator/session_service.py`
- Humanoid PPO evaluation:
  - `src/rewardlab/experiments/gymnasium_runner.py`

## Active Docs Updated

- `README.md`
- `specs/003-real-experiment-readiness/quickstart.md`
- `specs/003-real-experiment-readiness/verification-report.md`
- `specs/003-real-experiment-readiness/tasks.md`

These now explain that real backend execution and real reward iteration are
separate toggles.

## Remaining Blocker

The repo is now ready for a true token-consuming reward search, but that run has
not happened yet.

Blocked pending explicit user approval for paid API usage:

- use `OPENAI_API_KEY`
- enable `REWARDLAB_REWARD_DESIGN_MODE=openai`

Do not start the paid run until the user explicitly approves using the key.

## Recommended Next Step

Run a real multi-iteration Humanoid search with both real execution and real
reward generation enabled.

### Recommended Environment Variables

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
$env:REWARDLAB_REWARD_DESIGN_MODE = "openai"
$env:REWARDLAB_REWARD_DESIGN_MODEL = "gpt-5-mini"
$env:REWARDLAB_REWARD_DESIGN_REASONING_EFFORT = "medium"
$env:REWARDLAB_REWARD_DESIGN_MAX_TOKENS = "2000"
$env:REWARDLAB_PPO_TOTAL_TIMESTEPS = "50000"
$env:REWARDLAB_PPO_EVAL_RUNS = "5"
$env:REWARDLAB_PPO_CHECKPOINT_COUNT = "10"
```

### Suggested Start Command

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/humanoid_run.txt `
  --baseline-reward-file tools/fixtures/rewards/humanoid_baseline.py `
  --environment-id Humanoid-v4 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --json
```

Then run:

```powershell
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
```

Repeat `session step` until the session completes or pauses. Afterward:

```powershell
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
```

## Expected Comparison Target

The first real token-consuming Humanoid search should be compared against the
current real PPO baseline:

- `fitness_metric_mean = 0.482898`

## If The Next Run Pauses

Check session metadata and events for:

- `last_failed_design_error`
- `session.paused` event payload

Common causes will be:

- missing `OPENAI_API_KEY`
- invalid JSON from the model
- invalid reward code
- unsupported callable parameters introduced by the model

## Do Not Forget

- Do not use machine-level installs.
- Do not assume the presence of credentials is permission to spend money.
- Keep all artifacts inside this worktree.
- The next meaningful milestone is not another fixed-reward PPO run; it is a
  real multi-iteration Humanoid reward search using the OpenAI-backed designer.
