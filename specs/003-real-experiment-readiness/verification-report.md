# Verification Report: Real Experiment Readiness

## 2026-04-02

### Approved Install Step

Executed inside the active worktree `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install gymnasium
```

Detected versions:

- `gymnasium==1.2.3`
- `numpy==2.4.4`
- `cloudpickle==3.1.2`
- `Farama-Notifications==0.0.4`

### Real Gymnasium Smoke

Command shape executed:

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
  --session-id session-real-gym-smoke-20260402 `
  --json
.\.venv\Scripts\rewardlab.exe session step `
  --session-id session-real-gym-smoke-20260402 `
  --json
.\.venv\Scripts\rewardlab.exe session stop `
  --session-id session-real-gym-smoke-20260402 `
  --json
```

Observed evidence:

- Session id: `session-real-gym-smoke-20260402`
- Candidate from actual step: `session-real-gym-smoke-20260402-candidate-001`
- Iteration index from actual step: `1`
- Run artifact directory: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-smoke-gym\runs\session-real-gym-smoke-20260402-run-001`
- Manifest: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-smoke-gym\runs\session-real-gym-smoke-20260402-run-001\manifest.json`
- Metrics: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-smoke-gym\runs\session-real-gym-smoke-20260402-run-001\metrics.json`
- Report: `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\.rewardlab-smoke-gym\reports\session-real-gym-smoke-20260402.report.json`
- Manifest execution mode: `actual_backend`
- Episode reward: `388.244775`
- Step count: `200`

Result:

- PASS: Real Gymnasium session lifecycle completed through `session start`,
  `session step`, and `session stop`.
- PASS: The persisted run emitted non-empty metrics and artifact references.
- PASS: The exported report includes two iterations and references the actual
  backend run evidence.

### Remaining Blockers

- Isaac runtime install and smoke validation are still pending.

### Offline And Wrapper Validation

The following local validation commands were executed after the US2, US3, and
quality-wrapper changes landed in the worktree:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1
powershell -ExecutionPolicy Bypass -File tools\quality\run_real_backend_smokes.ps1
```

Observed evidence:

- `run_full_validation.ps1`: PASS
- Offline suite result: `63 passed, 2 skipped`
- `run_real_backend_smokes.ps1`: PASS
- Real Gymnasium wrapper smoke result: `1 passed`

Current status:

- PASS: Real robustness assessments and feedback artifact attachment now have
  focused integration coverage and pass in the worktree.
- PASS: Isaac actual-backend runtime handling, session persistence, and CLI
  smoke scaffolding now have offline-safe coverage and pass in the worktree.
- PASS: Quality scripts now keep pytest temp files inside the worktree and can
  optionally invoke real backend smokes.
- BLOCKED: Real Isaac smoke evidence still requires approval-gated runtime
  installation plus local `REWARDLAB_ISAAC_ENV_FACTORY` configuration.
