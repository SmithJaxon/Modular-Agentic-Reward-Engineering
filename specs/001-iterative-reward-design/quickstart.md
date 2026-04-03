# Quickstart: LLM-Guided Reward Function Iteration

## 0. Current Implementation Scope (2026-04-02)

- Implemented and verified: setup + foundational infrastructure + US1 iterative
  session lifecycle + US2 robustness and reward-hacking detection + US3 human
  and peer feedback gating
- Next target: Phase 6 polish and runtime validation evidence capture

## 1. Prerequisites

- Python 3.12
- Virtual environment tooling (`venv`)
- OpenAI API key available as environment variable
- Optional CUDA-capable GPU for faster experiment cycles

Set environment variable:

```powershell
$env:OPENAI_API_KEY="<your-api-key>"
```

## 2. Install and Validate Tooling

```powershell
venv\Scripts\python.exe -m pip install -e .[dev]
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
```

## 3. Run Contract and Schema Verification

```powershell
venv\Scripts\python.exe -m pytest tests/contract/test_session_lifecycle_cli.py -q
venv\Scripts\python.exe -m pytest tests/contract/test_backend_adapters.py -q
venv\Scripts\python.exe -m pytest tests/contract/test_feedback_cli.py -q
venv\Scripts\python.exe -m pytest tests/unit/test_foundational_components.py -q
venv\Scripts\python.exe tools/quality/validate_contracts.py
```

Expected result:
- CLI contracts and JSON schemas parse successfully.
- Invalid session configs are rejected by schema validation tests.

## 4. Run Deterministic Fixture Experiments and Robustness Checks

```powershell
venv\Scripts\python.exe -m pytest tests/integration/test_backend_selection.py -q
venv\Scripts\python.exe -m pytest tests/integration/test_reward_hack_probes.py -q
venv\Scripts\python.exe -m pytest tests/integration/test_iteration_loop.py -q
venv\Scripts\python.exe -m pytest tests/integration/test_feedback_gating.py -q
venv\Scripts\python.exe -m pytest tests/integration/test_feedback_conflicts.py -q
venv\Scripts\python.exe -m pytest tests/e2e/test_interrupt_best_candidate.py -q
```

Expected result:
- Iterative loop advances candidate versions and records reflections.
- Each `session step` routes through the configured backend adapter and executes
  robustness probes from `tools/reward_hack_probes/probe_matrix.yaml`.
- Reports include per-iteration risk levels and selection rationale when a
  minor robustness risk is accepted.
- Interrupt flow exports best-known candidate evidence to a report artifact.

## 5. Start a Session Manually

```powershell
rewardlab session start \
  --objective-file tools/fixtures/objectives/cartpole.txt \
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py \
  --environment-id cartpole-v1 \
  --environment-backend gymnasium \
  --no-improve-limit 3 \
  --max-iterations 20 \
  --feedback-gate one_required \
  --json
```

Isaac Gym variant using caller-provided files:

```powershell
rewardlab session start \
  --objective-file <path-to-isaac-objective-file> \
  --baseline-reward-file <path-to-isaac-baseline-file> \
  --environment-id isaac-ant-v0 \
  --environment-backend isaacgym \
  --no-improve-limit 3 \
  --max-iterations 20 \
  --feedback-gate one_required \
  --json
```

The repository only ships CartPole fixtures by default; provide Isaac-specific
objective and baseline files when exercising the Isaac Gym path manually.

Then run one step:

```powershell
rewardlab session step --session-id <SESSION_ID> --json
```

Attach human feedback for the returned candidate:

```powershell
rewardlab feedback submit-human `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --comment "Observed stable recovery after perturbations." `
  --score 0.20 `
  --artifact-ref artifacts/cartpole-demo.mp4 `
  --json
```

Request isolated peer feedback for the same candidate:

```powershell
rewardlab feedback request-peer `
  --session-id <SESSION_ID> `
  --candidate-id <CANDIDATE_ID> `
  --json
```

Then export the current report:

```powershell
rewardlab session report --session-id <SESSION_ID> --json
```

Expected result:
- Report iteration entries include feedback counts and compact feedback summaries.
- When `feedback_gate` is `one_required` or `both_required`, the exported report
  marks the final recommendation as pending until the selected candidate has the
  required review channels.

## 6. Validate Pause/Resume and Interrupt Recovery

```powershell
rewardlab session pause --session-id <SESSION_ID>
rewardlab session resume --session-id <SESSION_ID>
rewardlab session stop --session-id <SESSION_ID> --json
```

Expected result:
- Best-known candidate and evidence are preserved on stop.
- Report exports include robustness risk summaries for each completed candidate.
- Paused sessions resume without losing iteration history.

## 7. Quality Feedback Loop Before Merge

Run full verification:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests/unit tests/contract tests/integration tests/e2e -q -p no:cacheprovider -p no:tmpdir
venv\Scripts\python.exe tools/quality/check_headers.py src/rewardlab tests tools
```

Mandatory review checks:
- File headers are present and updated in touched Python files.
- Function/method headers cover all non-trivial routines.
- Dead code cleanup pass completed and verified in code review.

## 8. Phase 6 Development Bootstrap

Phase 5 is implemented. Use this baseline before Phase 6 work:

```powershell
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests/unit/test_foundational_components.py tests/contract/test_session_lifecycle_cli.py tests/contract/test_backend_adapters.py tests/contract/test_feedback_cli.py tests/integration/test_backend_selection.py tests/integration/test_reward_hack_probes.py tests/integration/test_iteration_loop.py tests/integration/test_feedback_gating.py tests/integration/test_feedback_conflicts.py tests/e2e/test_interrupt_best_candidate.py -q -p no:cacheprovider -p no:tmpdir
```

Then move into the Phase 6 polish and runtime-validation tasks in:
`specs/001-iterative-reward-design/tasks.md` and
`specs/001-iterative-reward-design/phase6-handoff.md`.
