# Quickstart: LLM-Guided Reward Function Iteration

## 1. Prerequisites

- Python 3.12
- Virtual environment tooling (`venv`)
- Optional OpenAI API key only if you want live peer-feedback calls
- Optional CUDA-capable GPU for heavier external experiment backends

Most local validation does not require an API key. Without `OPENAI_API_KEY`,
peer feedback falls back to a deterministic offline response.

If you need live peer feedback later, populate `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
# Then edit `.env` and set OPENAI_API_KEY=<your-api-key>
```

## 2. Install and Validate Tooling

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\tools\quality\run_full_validation.ps1
```

Expected result:
- Contract documents and JSON schemas parse successfully.
- Header/docstring audit returns no findings.
- `ruff`, `mypy`, and the full pytest suite complete successfully.

## 3. Start a Deterministic Local Session

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id cartpole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --json
```

Expected result:
- The command returns JSON containing `session_id`, `status`, and `created_at`.
- RewardLab creates a local `.rewardlab/` runtime directory if it does not exist.

## 4. Execute Iterations and Inspect Progress

```powershell
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
```

Expected result:
- Each step returns the next `iteration_index`, the new `candidate_id`, and the
  current `best_candidate_id`.
- Candidates, reflections, checkpoints, and events accumulate under `.rewardlab/`.

## 5. Pause, Resume, Stop, and Export a Report

```powershell
.\.venv\Scripts\rewardlab.exe session pause --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session resume --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
.\.venv\Scripts\rewardlab.exe session report --session-id <SESSION_ID> --json
```

Expected result:
- Pause and resume update session state without losing history.
- Stop exports the best-known candidate and returns `report_path`.
- `session report` writes or rewrites the JSON report for a non-running session.
- Paused sessions resume without losing iteration history.

## 6. Submit Human and Peer Feedback

Start a second session if you want feedback gating in the final report:

```powershell
.\.venv\Scripts\rewardlab.exe session start `
  --objective-file tools/fixtures/objectives/cartpole.txt `
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py `
  --environment-id cartpole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate one_required `
  --json
.\.venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
```

Attach human review:

```powershell
.\.venv\Scripts\rewardlab.exe feedback submit-human `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --comment "Stable balance and centered motion." `
  --score 0.8 `
  --artifact-ref demo.md `
  --json
```

Request peer review:

```powershell
.\.venv\Scripts\rewardlab.exe feedback request-peer `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --json
```

Expected result:
- Human feedback creates a bundle beneath `.rewardlab/reports/feedback_artifacts/`.
- Peer feedback works offline by default and uses the live model only when
  `OPENAI_API_KEY` is present.
- Final report summaries reflect the configured feedback gate.

## 7. Backend Routing Notes

- `gymnasium` workflows are fully runnable with the local deterministic fixtures.
- `isaacgym` routing is implemented and covered by contract/integration tests.
- For manual `isaacgym` runs, supply your own objective and reward files plus a
  valid environment identifier available in your local backend setup.

## 8. Quality Feedback Loop Before Merge

Run full verification:

```powershell
.\tools\quality\run_full_validation.ps1
```

Mandatory review checks:
- File headers are present and updated in touched Python files.
- Function/method headers cover all non-trivial routines.
- Dead code cleanup pass completed and verified in code review.
- Verification evidence is recorded in `specs/001-iterative-reward-design/verification-report.md`.
