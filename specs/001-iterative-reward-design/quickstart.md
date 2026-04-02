# Quickstart: LLM-Guided Reward Function Iteration

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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
ruff check .
mypy src
```

## 3. Run Contract and Schema Verification

```powershell
python -m pytest tests/contract -q
python -m pytest tests/unit/test_schema_contracts.py -q
```

Expected result:
- CLI contracts and JSON schemas parse successfully.
- Invalid session configs are rejected by schema validation tests.

## 4. Run Deterministic Fixture Experiments

```powershell
python -m pytest tests/integration/test_iteration_loop.py -q
python -m pytest tests/integration/test_reward_hack_probes.py -q
```

Expected result:
- Iterative loop advances candidate versions and records reflections.
- Robustness probes detect intentionally exploit-prone rewards.

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

Isaac Gym variant:

```powershell
rewardlab session start \
  --objective-file tools/fixtures/objectives/isaac_ant.txt \
  --baseline-reward-file tools/fixtures/rewards/isaac_ant_baseline.py \
  --environment-id isaac-ant-v0 \
  --environment-backend isaacgym \
  --no-improve-limit 3 \
  --max-iterations 20 \
  --feedback-gate one_required \
  --json
```

Then run one step:

```powershell
rewardlab session step --session-id <SESSION_ID> --json
```

## 6. Validate Pause/Resume and Interrupt Recovery

```powershell
rewardlab session pause --session-id <SESSION_ID>
rewardlab session resume --session-id <SESSION_ID>
rewardlab session stop --session-id <SESSION_ID> --json
```

Expected result:
- Best-known candidate and evidence are preserved on stop.
- Paused sessions resume without losing iteration history.

## 7. Quality Feedback Loop Before Merge

Run full verification:

```powershell
ruff check .
mypy src
python -m pytest tests/unit tests/contract tests/integration tests/e2e -q
python -m pytest tests/e2e/test_interrupt_best_candidate.py -q
```

Mandatory review checks:
- File headers are present and updated in touched Python files.
- Function/method headers cover all non-trivial routines.
- Dead code cleanup pass completed and verified in code review.
