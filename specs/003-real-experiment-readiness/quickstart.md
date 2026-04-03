# Quickstart: Real Experiment Readiness

This feature extends RewardLab from the current offline-safe MVP to actual
backend execution. The next agent should follow these steps while preserving the
existing worktree safety rules.

## Guardrails

- Work only in `C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass`
- Do not install anything without user approval
- Keep all installs inside `.venv`
- Do not modify PATH, registry, shell profiles, or anything outside the worktree
- Prefer offline tests first, then backend smokes only after approved setup

## Recommended Execution Order

1. Complete all code and tests that do not require new downloads.
2. When real backend work is blocked on missing packages, ask for approval and
   install only into `.venv`.
3. Finish Gymnasium real-run support before Isaac.
4. Run backend smoke validation and capture evidence in a verification report.

## Expected Approval-Gated Install Step

The next agent should ask before running something in one of these shapes:

```powershell
.\.venv\Scripts\python -m pip install -e .[dev,gymnasium,torch]
.\.venv\Scripts\python -m pip install <approved isaac runtime packages>
```

Record the exact approved command and the detected package versions here after
approval is granted. Do not pre-emptively install anything outside that gate.

Approved real Gymnasium install completed on 2026-04-02 with:

```powershell
.\.venv\Scripts\python.exe -m pip install gymnasium
```

Detected versions after install:

- `gymnasium==1.2.3`
- `numpy==2.4.4`
- `cloudpickle==3.1.2`
- `Farama-Notifications==0.0.4`

`torch` and any Isaac-specific runtime remain pending and should be handled as
separate approval-gated steps once the Isaac path is ready to validate.

After the approved Isaac runtime is installed, configure RewardLab with a
worktree-local callable that can construct the approved environment:

```powershell
$env:REWARDLAB_ISAAC_ENV_FACTORY = "your_module.your_isaac_wrapper:create_environment"
```

Optional preflight validation for `environment_id` can also be provided:

```powershell
$env:REWARDLAB_ISAAC_ENV_VALIDATOR = "your_module.your_isaac_wrapper:validate_environment"
```

The callable must stay inside the worktree and return an object with
`reset()`, `step()`, and `close()` methods compatible with the shared execution
runner.

## Backend-Specific Test Selection

Default pytest runs must stay offline-safe and therefore skip the real backend
smokes. Use the opt-in flags only after the corresponding runtime is available
inside `.venv`:

```powershell
.\.venv\Scripts\python -m pytest --run-real-gymnasium
.\.venv\Scripts\python -m pytest --run-real-isaacgym
```

Actual backend session stepping is also opt-in. Set the execution mode before
running the real Gymnasium or Isaac CLI flows:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
```

## Target Validation Commands

Offline regression:

```powershell
.\tools\quality\run_full_validation.ps1
```

Gymnasium real smoke target after implementation:

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

Isaac real smoke target after implementation:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
$env:REWARDLAB_ISAAC_ENV_FACTORY = "your_module.your_isaac_wrapper:create_environment"
$env:REWARDLAB_TEST_ISAAC_ENV_ID = "<APPROVED_ISAAC_ENV>"
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id $env:REWARDLAB_TEST_ISAAC_ENV_ID `
  --environment-backend isaacgym `
  --no-improve-limit 1 `
  --max-iterations 1 `
  --feedback-gate none `
  --json
```

If `isaacgym` is importable but the factory is missing, the session should
pause with an actionable error pointing to `REWARDLAB_ISAAC_ENV_FACTORY`
instead of silently falling back to an offline path.

Use `tools/fixtures/experiments/gymnasium_cartpole.json` and
`tools/fixtures/experiments/isaac_default.json` as the checked-in starting
configs for these runs.

## Completion Evidence

Real readiness is complete only when the next agent can point to:

- one real Gymnasium session report with non-empty run metrics and artifact refs
- one real Isaac session report with non-empty run metrics and artifact refs
- offline suite still passing
- exact install/setup commands recorded for reproducibility
