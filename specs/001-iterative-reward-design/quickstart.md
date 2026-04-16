# Quickstart: LLM-Guided Reward Function Iteration

## 0. Current Implementation Scope (2026-04-06)

- Implemented and verified: setup + foundational infrastructure + US1 iterative
  session lifecycle + US2 robustness and reward-hacking detection + US3 human
  and peer feedback gating + Phase 6 validation runner, regression tests, and
  runtime-gated smoke coverage
- Deterministic validation, Gymnasium runtime smoke, the Gymnasium CLI session
  workflow, and the new PPO-backed Gymnasium execution mode were validated on
  2026-04-06 in the project-scoped `venv`
- Adaptive PPO budget management, `budget_cap` session termination, and
  `remaining_budget` step responses were also validated on 2026-04-06
- Real Gymnasium and Isaac Gym smoke checks run only when those optional
  runtimes are installed on a supported machine
- Live OpenAI smoke validation is opt-in, passed on 2026-04-02 with the
  repo-local `.env` path, and still requires a valid `OPENAI_API_KEY`
- `.venv-mujoco` now validates a real `Humanoid-v4` session step using the new
  Gymnasium PPO engine
- Agentic tool-calling rework now has an executable Phase 0-1 scaffold under
  `rewardlab agent ...`; real experiment tools are still being migrated

## 1. Prerequisites

- Python 3.12
- Virtual environment tooling (`venv`)
- OpenAI API key available in the project `.env` or the current environment
- Optional CUDA-capable GPU for faster experiment cycles
- Optional local RL runtimes:
  - Gymnasium via `venv\Scripts\python.exe -m pip install -e .[dev,rl]`
  - Isaac Gym via a separate compatible vendor/runtime installation
- Optional live OpenAI smoke configuration:
  - `REWARDLAB_OPENAI_SMOKE_MODEL=gpt-4o-mini`

Preferred local credential setup:

```powershell
Copy-Item .env.example .env
```

Then populate `OPENAI_API_KEY` in the project `.env`. The OpenAI client and the
validation runner auto-load that file when the key is not already set in the
shell.

## 2. Install and Validate Tooling

```powershell
venv\Scripts\python.exe -m pip install -e .[dev]
venv\Scripts\python.exe -m ruff check src tests tools
venv\Scripts\python.exe -m mypy src
```

Install the optional Gymnasium runtime before executing the Gymnasium smoke
suite:

```powershell
venv\Scripts\python.exe -m pip install -e .[dev,rl]
```

Current validated local stack on this workstation:
- Editable `rewardlab` install in the project `venv`
- `gymnasium 0.29.1`
- `stable-baselines3 2.8.0`
- `torch 2.11.0`

Install the optional OpenAI dependency before executing the live OpenAI smoke
test:

```powershell
venv\Scripts\python.exe -m pip install -e .[llm]
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

Run a real PPO-backed Gymnasium iteration instead of the deterministic adapter:

```powershell
rewardlab session start \
  --objective-file tools/fixtures/objectives/cartpole.txt \
  --baseline-reward-file tools/fixtures/rewards/cartpole_baseline.py \
  --environment-id CartPole-v1 \
  --environment-backend gymnasium \
  --no-improve-limit 3 \
  --max-iterations 5 \
  --feedback-gate none \
  --execution-mode ppo \
  --budget-mode adaptive \
  --llm-provider openai \
  --total-training-timesteps 50000 \
  --total-evaluation-episodes 40 \
  --max-llm-calls 4 \
  --target-reflection-checkpoints 4 \
  --ppo-num-envs 1 \
  --ppo-n-steps 128 \
  --ppo-batch-size 128 \
  --json
```

In PPO mode, the agent now manages the session budget on its own. The
session-level budget controls the whole search, while `--ppo-total-timesteps`,
`--evaluation-episodes`, `--reflection-episodes`, and related PPO knobs remain
available as legacy fixed-mode settings and as ceilings/defaults for adaptive
planning.

When you execute:

```powershell
rewardlab session step --session-id <SESSION_ID> --json
```

the response now includes `remaining_budget`. Adaptive PPO sessions may stop
early with `status=completed` and `stop_reason=budget_cap` when the planner
cannot afford another full iteration.

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

The repository currently ships CartPole and Humanoid fixtures by default.
Provide Isaac-specific files when exercising the Isaac Gym path manually.

For MuJoCo-backed environments such as `Humanoid-v4`, run the same command
pattern from `.venv-mujoco\Scripts\Activate.ps1` and swap in
`--environment-id Humanoid-v4`. A practical starting point for a longer
Humanoid search on this machine is:

```powershell
.venv-mujoco\Scripts\rewardlab.exe session start `
  --objective-file tools\fixtures\objectives\humanoid.txt `
  --baseline-reward-file tools\fixtures\rewards\humanoid_baseline.py `
  --environment-id Humanoid-v4 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --execution-mode ppo `
  --budget-mode adaptive `
  --llm-provider openai `
  --total-training-timesteps 200000 `
  --total-evaluation-episodes 40 `
  --max-llm-calls 4 `
  --target-reflection-checkpoints 4 `
  --ppo-num-envs 1 `
  --ppo-n-steps 128 `
  --ppo-batch-size 128 `
  --json
```

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
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -DeterministicOnly
```

Mandatory review checks:
- File headers are present and updated in touched Python files.
- Function/method headers cover all non-trivial routines.
- Dead code cleanup pass completed and verified in code review.

## 8. One-Command Validation Runner

Run deterministic validation only:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -DeterministicOnly
```

Run deterministic validation plus any runtime suites that are available:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1
```

Require both runtime suites on a supported machine:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireRuntimeSuites
```

Expected behavior:
- The deterministic suite always runs.
- `tests/integration/test_gymnasium_runtime.py` only executes when `gymnasium`
  is installed.
- `tests/integration/test_isaacgym_runtime.py` only executes when `isaacgym` is
  installed.
- `-RequireRuntimeSuites` converts missing optional runtimes into a validation
  failure so supported machines can capture strict runtime evidence.

Run the live OpenAI client smoke test explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -EnableOpenAISmoke
$env:REWARDLAB_OPENAI_SMOKE_MODEL="gpt-4o-mini"
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireOpenAISmoke
```

Expected behavior:
- The runner auto-loads `OPENAI_API_KEY` from the project `.env` when the key
  is not already exported.
- The runner sets `REWARDLAB_ENABLE_OPENAI_LIVE_TESTS=1` for the smoke command,
  so the opt-in flag only needs to be exported when invoking the test directly.
- The test requires the optional `openai` dependency and a valid
  `OPENAI_API_KEY`.
- `REWARDLAB_OPENAI_SMOKE_MODEL` can override the default smoke model.
- Authentication failures indicate a credential/configuration issue rather than
  a deterministic orchestration regression.

Validated Gymnasium CLI workflow on 2026-04-02:

```powershell
New-Item -ItemType Directory -Force .tmp-gymnasium-ready | Out-Null
$env:REWARDLAB_DATA_DIR=(Resolve-Path '.tmp-gymnasium-ready').Path
venv\Scripts\rewardlab.exe session start --objective-file tools\fixtures\objectives\cartpole.txt --baseline-reward-file tools\fixtures\rewards\cartpole_baseline.py --environment-id cartpole-v1 --environment-backend gymnasium --no-improve-limit 3 --max-iterations 5 --feedback-gate none --json
venv\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
venv\Scripts\rewardlab.exe session report --session-id <SESSION_ID> --json
venv\Scripts\rewardlab.exe session stop --session-id <SESSION_ID> --json
```

Expected result:
- Session artifacts are written under `REWARDLAB_DATA_DIR`, including
  `rewardlab.sqlite3`, `events.jsonl`, `checkpoints/`, and `reports/`.
