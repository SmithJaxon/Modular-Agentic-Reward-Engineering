# RewardLab

RewardLab is a local-first Python CLI for iterative reward-function design.
It provides deterministic session orchestration, backend routing, robustness
assessment hooks, feedback gating, and interruption-safe report export.

## What It Does

- Starts and manages reward-optimization sessions from an objective file and
  baseline reward definition.
- Runs a deterministic MVP iteration loop that evaluates, reflects, revises,
  and re-ranks reward candidates.
- Routes experiment execution through `gymnasium` or `isaacgym` backends.
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
.\.venv\Scripts\python -m pip install <approved isaac runtime packages>
```

Default `pytest` runs skip the approval-gated real backend smokes. Opt in only
after the required runtime is present:

```powershell
.\.venv\Scripts\python -m pytest --run-real-gymnasium
.\.venv\Scripts\python -m pytest --run-real-isaacgym
```

To route `session step` through the real backend path instead of the offline
heuristic path, set:

```powershell
$env:REWARDLAB_EXECUTION_MODE = "actual_backend"
```

Use `offline_test` to force the deterministic validation mode.

## Offline Workflow

Most validation and CLI flows run without an API key. The peer-feedback path
uses a deterministic local fallback unless `OPENAI_API_KEY` is set.

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
`gpt-5-nano` model with a single short critique prompt. If no key is present,
RewardLab stays offline and uses the deterministic fallback path instead.
RewardLab auto-loads the nearest local `.env` file from the current working
directory upward, while process environment variables still take precedence.

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

The real-backend smoke wrapper defaults to the Gymnasium smoke and can also run
the Isaac smoke once the approved runtime is installed and
`REWARDLAB_ISAAC_ENV_FACTORY` plus `REWARDLAB_TEST_ISAAC_ENV_ID` are set.

Detailed operator steps and manual workflow examples are in
`specs/001-iterative-reward-design/quickstart.md` and
`specs/003-real-experiment-readiness/quickstart.md`.
