# Verification Report: Real Experiment Readiness

## 2026-04-02

### Approved Gymnasium Install

```powershell
.\.venv\Scripts\python.exe -m pip install gymnasium
```

Detected versions:

- `gymnasium==1.2.3`
- `numpy==2.4.4`
- `cloudpickle==3.1.2`
- `Farama-Notifications==0.0.4`

### Real Gymnasium Smoke

Observed evidence:

- real `CartPole-v1` session lifecycle completed in `actual_backend` mode
- persisted manifest, metrics, and report files were written under the worktree
- report evidence clearly referenced the actual run artifacts

Status:

- PASS: Gymnasium smoke path is working
- PASS: robustness and artifact plumbing remain integrated on the Gymnasium path

## 2026-04-06

### Scope Reset

Observed changes:

- Isaac-specific runtime branches, tests, fixtures, schemas, and tooling were
  removed from the active worktree
- contracts now validate only the `gymnasium` backend
- Humanoid fixtures were added for PPO-based evaluation

### Humanoid PPO Implementation

Observed code status:

- `gymnasium_runner.py` now contains a Humanoid PPO protocol for
  `Humanoid-v4` and `Humanoid-v5`
- final score shape matches the EUREKA-style protocol:
  - 5 PPO runs
  - 10 checkpoints per run
  - average of the per-run best checkpoint mean `x_velocity`
- missing `stable_baselines3` now produces an explicit prerequisite error

### Local Validation

Validated in this worktree:

- `ruff` on the touched Gymnasium/runtime files: PASS
- targeted mypy on the touched runtime files: PASS
- targeted `pytest` for `tests/integration/test_gymnasium_real_experiment.py`: PASS

Known local validation wrinkle:

- combined multi-file pytest slices can still hit a Windows basetemp cleanup
  permission issue in this environment when the temp root is reused
- this is separate from the Gymnasium runtime logic; individual targeted files
  pass when run with a fresh worktree-local temp path

### Remaining Blocker

- BLOCKED at that time: real `Humanoid-v4` validation still required user
  approval to install `stable-baselines3` inside `.venv`

## 2026-04-06 Humanoid PPO Run

### Approved PPO Install

Executed inside the active worktree `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[ppo]
```

Detected versions:

- `stable_baselines3==2.8.0`
- `pandas==3.0.2`

### Actual Humanoid Session

Session:

- `session-real-humanoid-ppo-20260406`
- environment: `Humanoid-v4`
- backend: `gymnasium`
- execution mode: `actual_backend`

Persisted evidence:

- Manifest:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\runs\session-real-humanoid-ppo-20260406-run-001\manifest.json`
- Metrics:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\runs\session-real-humanoid-ppo-20260406-run-001\metrics.json`
- Report:
  `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-humanoid-run\reports\session-real-humanoid-ppo-20260406.report.json`

Observed metrics:

- `fitness_metric_name`: `mean_x_velocity`
- `fitness_metric_mean`: `0.482898`
- `per_run_best_mean_x_velocity`:
  - `0.577095`
  - `0.441847`
  - `0.515695`
  - `0.531256`
  - `0.348597`
- `train_timesteps`: `50000`
- `checkpoint_count`: `10`
- `evaluation_run_count`: `5`
- `evaluation_protocol`: `humanoid_ppo_max_checkpoint_mean_x_velocity`

Implementation note:

- The first Humanoid attempt exposed an SB3 environment-compatibility issue in
  the reward-shaping wrapper.
- After patching the wrapper to be a real Gymnasium `Env`, the rerun completed
  successfully and produced the evidence above.

Status:

- PASS: actual Gymnasium Humanoid PPO evaluation now runs end to end
- PASS: persisted artifacts and report evidence exist for the full pipeline path

## 2026-04-06 OpenAI Reward Iteration

### Implementation

Observed code status:

- `src/rewardlab/orchestrator/reward_designer.py` now provides two explicit
  reward-designer modes:
  - `deterministic`
  - `openai`
- `SessionService` now records the active reward-designer mode in session
  metadata and routes candidate generation through the configured designer
- actual-backend iteration now passes the latest reflection and latest run
  metrics into the next reward-design request
- OpenAI-backed reward generation is opt-in through
  `REWARDLAB_REWARD_DESIGN_MODE=openai`
- invalid model output, unsupported callable parameters, or missing credentials
  now pause the session with an explicit design error instead of silently
  producing another deterministic comment-only revision

### Local Validation

Validated in this worktree:

- `ruff` on the new reward-designer and session files: PASS
- targeted mypy on the new reward-designer and session files: PASS
- targeted `pytest` on:
  - `tests/unit/test_reward_designer.py`
  - `tests/integration/test_gymnasium_real_experiment.py`
  - Result: `10 passed`

Known local validation wrinkle:

- broader pytest slices that rely on `tmp_path` still hit the same Windows
  basetemp cleanup permission bug already observed elsewhere in this worktree
- the cleanup failure happens during pytest teardown and is separate from the
  reward-designer implementation itself

### Remaining Blocker

- BLOCKED: a paid multi-iteration Humanoid reward search has not been run yet
  because that now correctly requires explicit user approval to use
  `OPENAI_API_KEY` and enable `REWARDLAB_REWARD_DESIGN_MODE=openai`

## 2026-04-10 Follow-Up Status

### OpenAI-Backed Humanoid Execution

Observed status in this worktree:

- real `Humanoid-v4` sessions with `REWARDLAB_REWARD_DESIGN_MODE=openai` have
  been executed successfully
- reward-designer retry and signature-handling robustness improvements were
  added after earlier malformed-response pauses

### Clarification On Token Limits

- `REWARDLAB_REWARD_DESIGN_MAX_TOKENS` is a per-call completion cap
- for `gpt-5-mini`, per-call completion tokens are capped by the model
  (currently `128000`); higher values fail with API validation errors

### Tranche Status

- `003-real-experiment-readiness` is complete for Gymnasium-only runtime and
  Humanoid PPO execution
- follow-on architecture work now continues in
  `specs/004-agent-tool-calling-architecture/`
