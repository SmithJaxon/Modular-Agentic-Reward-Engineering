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

## Backend-Specific Test Selection

Default pytest runs must stay offline-safe and therefore skip the real backend
smokes. Use the opt-in flags only after the corresponding runtime is available
inside `.venv`:

```powershell
.\.venv\Scripts\python -m pytest --run-real-gymnasium
.\.venv\Scripts\python -m pytest --run-real-isaacgym
```

## Target Validation Commands

Offline regression:

```powershell
.\tools\quality\run_full_validation.ps1
```

Gymnasium real smoke target after implementation:

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id CartPole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 2 `
  --max-iterations 2 `
  --feedback-gate none `
  --json
```

Isaac real smoke target after implementation:

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id <APPROVED_ISAAC_ENV> `
  --environment-backend isaacgym `
  --no-improve-limit 1 `
  --max-iterations 1 `
  --feedback-gate none `
  --json
```

Use `tools/fixtures/experiments/gymnasium_cartpole.json` and
`tools/fixtures/experiments/isaac_default.json` as the checked-in starting
configs for these runs.

## Completion Evidence

Real readiness is complete only when the next agent can point to:

- one real Gymnasium session report with non-empty run metrics and artifact refs
- one real Isaac session report with non-empty run metrics and artifact refs
- offline suite still passing
- exact install/setup commands recorded for reproducibility
