# Phase 7 Pickup Handoff: Exact Resume Notes After Gymnasium PPO Follow-Up

**Prepared**: 2026-04-06  
**Feature**: `001-iterative-reward-design`  
**Branch**: `iterative-main`  
**Workspace state**: dirty working tree, not yet committed

## 1. Resume Point

This workspace now includes:

- the real Gymnasium PPO execution path
- OpenAI-backed reward synthesis/revision in that loop
- adaptive session-level budget management
- `budget_cap` stop handling
- `remaining_budget` step output
- local MuJoCo/Humanoid validation support in `.venv-mujoco`

The main remaining blocker is still external Isaac Gym validation on a supported
machine.

## 2. Current Blocking Items

- `T064` is still open because Isaac Gym runtime validation has not been run on
  a supported machine.
- Gymnasium/PPO work is no longer the blocker on this workstation.
- Comparison protocol work is still open if you want stronger Humanoid or Ant
  benchmarking before the Isaac Gym pass.

## 3. What Changed In The Latest Pass

Key implementation additions:

- `src/rewardlab/experiments/gymnasium_runtime.py`
- `src/rewardlab/orchestrator/budget_manager.py`
- `src/rewardlab/llm/reward_prompting.py`

Key integration updates:

- `src/rewardlab/orchestrator/iteration_engine.py`
- `src/rewardlab/orchestrator/session_service.py`
- `src/rewardlab/cli/session_commands.py`
- `src/rewardlab/experiments/robustness_runner.py`
- `src/rewardlab/schemas/session_report.py`

Key docs updated:

- `README.md`
- `specs/001-iterative-reward-design/quickstart.md`
- `specs/001-iterative-reward-design/verification-report.md`
- `specs/001-iterative-reward-design/phase7-handoff.md`

## 4. Validation Already Run

Passed on 2026-04-06:

- `venv\Scripts\python.exe -m ruff check src tests`
- `venv\Scripts\python.exe -m mypy src`
- `venv\Scripts\python.exe -m pytest tests\unit tests\contract tests\integration tests\e2e -q -p no:cacheprovider -p no:tmpdir --ignore tests\integration\test_isaacgym_runtime.py`
  - result: `53 passed, 1 skipped`
- `.venv-mujoco\Scripts\python.exe -m pytest tests\integration\test_gymnasium_ppo_iteration.py -q -p no:cacheprovider -p no:tmpdir`
  - result: `2 passed`

Earlier validations that still matter:

- live OpenAI smoke passed on 2026-04-02 through the repo-local `.env` path
- Gymnasium runtime smoke passed locally
- real local `Humanoid-v4` runtime and PPO session flow passed in `.venv-mujoco`

Not rerun after the adaptive-budget follow-up:

- `powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireOpenAISmoke`

## 5. Current Working Tree Snapshot

At handoff time, `git status --short` was:

```text
 M .env.example
 M README.md
 M pyproject.toml
 M specs/001-iterative-reward-design/checklists/requirements.md
 M specs/001-iterative-reward-design/contracts/session-report.schema.json
 M specs/001-iterative-reward-design/plan.md
 M specs/001-iterative-reward-design/quickstart.md
 M specs/001-iterative-reward-design/spec.md
 M specs/001-iterative-reward-design/tasks.md
 M src/rewardlab/cli/session_commands.py
 M src/rewardlab/experiments/backends/gymnasium_backend.py
 M src/rewardlab/experiments/robustness_runner.py
 M src/rewardlab/llm/openai_client.py
 M src/rewardlab/orchestrator/iteration_engine.py
 M src/rewardlab/orchestrator/session_service.py
 M src/rewardlab/persistence/session_repository.py
 M src/rewardlab/persistence/sqlite_store.py
 M src/rewardlab/schemas/session_report.py
 M src/rewardlab/selection/risk_analyzer.py
 M tests/integration/test_reward_hack_probes.py
 M tests/unit/test_foundational_components.py
?? .tmp-agent-review-phase6.md
?? specs/001-iterative-reward-design/phase7-handoff.md
?? specs/001-iterative-reward-design/phase7-pickup-handoff.md
?? specs/001-iterative-reward-design/verification-report.md
?? src/rewardlab/experiments/gymnasium_runtime.py
?? src/rewardlab/llm/reward_prompting.py
?? src/rewardlab/orchestrator/budget_manager.py
?? src/rewardlab/utils/env_loader.py
?? tests/e2e/test_iteration_cycle_budget.py
?? tests/integration/test_gymnasium_ppo_iteration.py
?? tests/integration/test_gymnasium_runtime.py
?? tests/integration/test_isaacgym_runtime.py
?? tests/integration/test_openai_runtime.py
?? tests/unit/test_env_loader.py
?? tests/unit/test_gymnasium_reward_program.py
?? tests/unit/test_header_audit.py
?? tests/unit/test_openai_client.py
?? tests/unit/test_schema_contracts.py
?? tools/quality/run_full_validation.ps1
```

Do not reset or clean this worktree unless explicitly asked.

## 6. Exact Next Steps

If staying on this machine:

1. Use `.venv-mujoco` for longer Humanoid or Ant experiments.
2. Start with the adaptive-budget CLI flow documented in:
   `specs/001-iterative-reward-design/quickstart.md`
3. Use `remaining_budget` plus exported reports to decide whether the current
   budget was enough.

If moving to the supported Isaac Gym machine:

1. Install RL dependencies:

```powershell
venv\Scripts\python.exe -m pip install -e .[dev,rl]
```

2. Ensure Isaac Gym vendor/runtime requirements are present.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireRuntimeSuites
```

4. Update status/evidence docs only after those runtime checks pass.

## 7. Resume Guidance

- Start from branch `iterative-main`.
- Treat `phase7-handoff.md` as the high-level plan.
- Treat this file as the exact recovery note for the current workspace.
- Keep `T064` honestly open until real Isaac Gym evidence exists.
