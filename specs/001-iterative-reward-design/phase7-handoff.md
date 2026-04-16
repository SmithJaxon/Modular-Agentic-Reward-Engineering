# Phase 7 Handoff: Gymnasium PPO Complete, Isaac Gym Still Pending

**Prepared**: 2026-04-06  
**Feature**: `001-iterative-reward-design`  
**Prepared from integration branch**: `iterative-main`  
**Target integration branch for next work**: `iterative-main`

## 1. Current State

The project is now in a materially different state from the original Phase 6
handoff:

- Core orchestration, persistence, reporting, robustness probes, and feedback
  gating are complete.
- The real Gymnasium PPO execution engine is implemented.
- Reward generation and revision can now use OpenAI in the Gymnasium PPO loop.
- PPO sessions now manage a session-level budget adaptively by default.
- `session step --json` now reports `remaining_budget`.
- Adaptive PPO sessions can terminate with `stop_reason="budget_cap"`.
- A real local MuJoCo `Humanoid-v4` path was validated in `.venv-mujoco`.
- `T064` is still open because Isaac Gym runtime validation has not been run on
  a supported machine.

## 2. What Is Fully Done

Completed implementation:

- `T001`-`T063`
- `T065`-`T067`
- Real Gymnasium PPO engine
- Adaptive session-level budget manager
- OpenAI-backed reward synthesis/revision in the Gymnasium loop
- MuJoCo/Humanoid local runtime validation path

Validated on 2026-04-06:

```powershell
venv\Scripts\python.exe -m ruff check src tests
venv\Scripts\python.exe -m mypy src
venv\Scripts\python.exe -m pytest tests\unit tests\contract tests\integration tests\e2e -q -p no:cacheprovider -p no:tmpdir --ignore tests\integration\test_isaacgym_runtime.py
.venv-mujoco\Scripts\python.exe -m pytest tests\integration\test_gymnasium_ppo_iteration.py -q -p no:cacheprovider -p no:tmpdir
```

Observed results:

- `ruff`: pass
- `mypy`: pass
- local non-Isaac pytest sweep: `53 passed, 1 skipped`
- MuJoCo PPO integration tests: `2 passed`

## 3. Remaining Work

### External blocker

- Close `T064` by capturing real Isaac Gym runtime evidence on a supported
  machine.

### Experimental follow-up

- Lock a comparison protocol for Humanoid/Ant-style experiments that is close
  enough to Eureka to be useful now, while reserving true Isaac Gym comparison
  work for the supported machine.
- Run longer adaptive-budget Gymnasium/MuJoCo experiments and compare resulting
  reward programs under fixed multi-seed budgets.

## 4. Important Files

Core Gymnasium execution path:

- `src/rewardlab/experiments/gymnasium_runtime.py`
- `src/rewardlab/experiments/backends/gymnasium_backend.py`
- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/orchestrator/budget_manager.py`
- `src/rewardlab/cli/session_commands.py`
- `src/rewardlab/llm/reward_prompting.py`
- `src/rewardlab/llm/openai_client.py`

Robustness and reporting:

- `src/rewardlab/experiments/robustness_runner.py`
- `src/rewardlab/selection/risk_analyzer.py`
- `src/rewardlab/orchestrator/reporting.py`
- `src/rewardlab/schemas/session_report.py`

Documentation and evidence:

- `README.md`
- `specs/001-iterative-reward-design/quickstart.md`
- `specs/001-iterative-reward-design/verification-report.md`
- `specs/001-iterative-reward-design/tasks.md`

## 5. Recommended Next Steps

### If continuing on this machine

Use `.venv-mujoco` and the adaptive budget flow for longer Gymnasium/MuJoCo
experiments. A practical Humanoid starting point is:

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

Then run:

```powershell
.venv-mujoco\Scripts\rewardlab.exe session step --session-id <SESSION_ID> --json
```

Watch the returned `remaining_budget` and exported report artifacts before
deciding whether to extend the comparison protocol or adjust budgets.

### If continuing on a supported Isaac Gym machine

1. Install project RL dependencies:

```powershell
venv\Scripts\python.exe -m pip install -e .[dev,rl]
```

2. Ensure Isaac Gym vendor/runtime requirements are present.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireRuntimeSuites
```

4. Update:

- `specs/001-iterative-reward-design/verification-report.md`
- `specs/001-iterative-reward-design/checklists/requirements.md`
- `specs/001-iterative-reward-design/spec.md`
- `specs/001-iterative-reward-design/plan.md`
- `specs/001-iterative-reward-design/tasks.md`
- `README.md`

Only mark `T064` complete after real Isaac Gym runtime evidence exists.

## 6. Guardrails

- Do not reset or clean the dirty worktree unless explicitly asked.
- Keep `T064` honestly open until supported-machine Isaac Gym evidence is
  captured.
- Treat Gymnasium/MuJoCo experiments and Isaac Gym closure as separate tracks:
  the first is usable now, the second still needs the other machine.
