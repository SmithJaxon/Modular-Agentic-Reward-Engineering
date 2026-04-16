# RewardLab: LLM-Guided Reward Function Iteration

This repository contains a modular Python pipeline for iteratively designing and
validating reinforcement-learning reward functions using LLM-guided reflection,
robustness checks, and human or peer feedback gates.

## Current Status (2026-04-06)

- Core orchestration, validation tooling, the real Gymnasium PPO execution
  engine, and the adaptive session-level budget manager are implemented
- The project-scoped `venv` passed `ruff`, `mypy`, and the full non-Isaac local
  pytest sweep on 2026-04-06 with `53 passed, 1 skipped`
- Deterministic validation, Gymnasium runtime smoke, live OpenAI smoke, and the
  Gymnasium CLI session workflow are green on this workstation
- The project now has a real Gymnasium PPO execution mode in addition to the
  fast deterministic adapter path, with local artifact capture, reflection
  summaries, and LLM reward revision suitable for iterative reward search
- PPO sessions now manage budget adaptively by default, return
  `remaining_budget` from `session step`, and can terminate with
  `stop_reason="budget_cap"`
- The local `venv` now contains the documented Gymnasium stack, including the
  editable `rewardlab` install, `gymnasium 0.29.1`, `stable-baselines3 2.8.0`,
  and `torch 2.11.0`
- The MuJoCo-capable `.venv-mujoco` also contains the PPO stack and validated a
  real `Humanoid-v4` session step plus adaptive-budget PPO integration tests on
  2026-04-06
- `T064` remains open because Isaac Gym runtime validation is still pending

## Agentic Rework Status (2026-04-11)

- Agentic tool-calling rework is now in progress with Phases 0-6 implemented:
  `rewardlab agent run/status/events/report` is implemented.
- The scaffold executes decision turns through a broker/registry path with
  active `run_experiment`, `run_probe_suite`, `compare_candidates`,
  `export_report`, `budget_snapshot`, and `read_artifact` tools.
- Tool execution now routes through isolated worker task packets with
  worker start/completion events in run traces.
- Stop guidance now runs through a dedicated module with objective, plateau,
  risk-ceiling, cost-efficiency, and hard-budget exhaustion checks.
- Decision policy now supports compare cadence controls and probe-first compare
  behavior (`min_candidates_before_compare`, `compare_every_new_candidates`,
  `require_probe_before_compare`).
- Primary planning now supports an OpenAI-driven mode (`planner_provider: openai`)
  where the primary agent decides tool calls from context-rich prompts, with
  heuristic fallback if planner output is unavailable.
- Planner budget accounting now tracks planner-call API input/output tokens and
  model-call usage in the same run budget ledger.
- Strict planner mode is available via `planner_fallback_enabled: false` to
  fail closed instead of falling back to heuristic decisions.
- Decision traces now include `decision_source` so runs can distinguish
  heuristic actions from `llm_openai` planner actions.
- Planner validation failures are now traceable in events
  (`planner.validation_failed`) and summarized in reports under
  `planner_feedback_summary`.
- `configs/agentic/humanoid_main_openai_strict.yaml` provides fail-closed
  planner mode (`planner_fallback_enabled: false`) for strict runs.
- Broker budget checks are now per-call estimate aware for compute tools so
  oversized `run_experiment` and `run_probe_suite` requests are rejected before
  execution.
- Real smoke validation confirmed end-to-end tool traces on 2026-04-11
  (`agentrun-0334541a482d`, `stop_reason=objective_met`).
- The existing `session start/step/report/stop` workflow remains the fully
  production-ready path for real PPO experiment execution.
- Architecture and run-spec references:
  - `specs/agentic-tool-calling-plan.md`
  - `configs/agentic/`

## Primary Docs

- Feature spec: `specs/001-iterative-reward-design/spec.md`
- Implementation plan: `specs/001-iterative-reward-design/plan.md`
- Task board: `specs/001-iterative-reward-design/tasks.md`
- Phase 6 handoff: `specs/001-iterative-reward-design/phase6-handoff.md`
- Next handoff: `specs/001-iterative-reward-design/phase7-handoff.md`
- Exact pickup handoff: `specs/001-iterative-reward-design/phase7-pickup-handoff.md`
- Operator quickstart: `specs/001-iterative-reward-design/quickstart.md`
- Verification evidence: `specs/001-iterative-reward-design/verification-report.md`
- Agentic rework plan (planning doc): `specs/agentic-tool-calling-plan.md`
- Agentic improvement backlog (living): `specs/agentic-improvement-backlog.md`
- Agentic run spec examples (planning inputs): `configs/agentic/README.md`

Historical handoff docs under `specs/001-iterative-reward-design/phase*.md` are
point-in-time records and may not describe the latest planned architecture
direction.

## Local Setup (Project-Scoped Venv)

```powershell
venv\Scripts\python.exe -m pip install -e .[dev]
```

Install optional RL dependencies before running Gymnasium runtime smoke checks:

```powershell
venv\Scripts\python.exe -m pip install -e .[dev,rl]
```

Install the optional OpenAI dependency before running the live OpenAI smoke
test:

```powershell
venv\Scripts\python.exe -m pip install -e .[llm]
```

For the live OpenAI path, copy `.env.example` to `.env` and set
`OPENAI_API_KEY` there. The project client and validation runner auto-load the
repo-local `.env` when the key is not already exported in the shell.

Isaac Gym validation requires a separate compatible vendor/runtime installation.
The live OpenAI smoke test requires the optional `openai` dependency and a valid
`OPENAI_API_KEY`. It optionally honors `REWARDLAB_OPENAI_SMOKE_MODEL`.

For MuJoCo environments such as `Humanoid-v4`, use the separate MuJoCo-capable
environment:

```powershell
.venv-mujoco\Scripts\python.exe -m pip install -e .[dev,llm,rl]
```

## Real Gymnasium PPO Mode

Use `rewardlab session start` with `--execution-mode ppo` to train a real PPO
policy against the candidate reward function instead of the deterministic local
simulator. The PPO path stores reward programs, summary metrics, reflection
text, and trained policy checkpoints under `REWARDLAB_DATA_DIR/experiments/`.
In PPO mode, RewardLab now manages an adaptive session-level budget by default:
set total search budget with `--total-training-timesteps`,
`--total-evaluation-episodes`, and `--max-llm-calls`, then let the agent decide
how much of that budget to spend per iteration.

Minimal CartPole example:

```powershell
venv\Scripts\rewardlab.exe session start `
  --objective-file tools\fixtures\objectives\cartpole.txt `
  --baseline-reward-file tools\fixtures\rewards\cartpole_baseline.py `
  --environment-id CartPole-v1 `
  --environment-backend gymnasium `
  --no-improve-limit 3 `
  --max-iterations 5 `
  --feedback-gate none `
  --execution-mode ppo `
  --budget-mode adaptive `
  --llm-provider openai `
  --total-training-timesteps 50000 `
  --total-evaluation-episodes 40 `
  --max-llm-calls 4 `
  --target-reflection-checkpoints 4 `
  --ppo-num-envs 1 `
  --ppo-n-steps 128 `
  --ppo-batch-size 128 `
  --json
```

Notes:

- `rewardlab session step --json` now includes a `remaining_budget` payload for
  training timesteps, evaluation episodes, and LLM calls.
- Adaptive PPO sessions may finish before `--max-iterations` when the planner
  can no longer fund another iteration; those runs export with
  `stop_reason="budget_cap"`.
- Use `--budget-mode fixed` if you want to keep manual per-iteration control
  with the low-level PPO knobs.

## Validation Commands

```powershell
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -DeterministicOnly
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireRuntimeSuites
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -EnableOpenAISmoke
$env:REWARDLAB_OPENAI_SMOKE_MODEL="gpt-4o-mini"
powershell -ExecutionPolicy Bypass -File tools\quality\run_full_validation.ps1 -RequireOpenAISmoke
```

The validation runner keeps deterministic checks separate from the optional
Gymnasium and Isaac Gym smoke suites. `-RequireRuntimeSuites` is intended for a
supported machine where those runtimes are expected to be installed. The live
OpenAI smoke test is opt-in, auto-loads `OPENAI_API_KEY` from the project
`.env` when needed, and requires a valid credential.

## Runtime Artifacts

By default the CLI writes session state under `.rewardlab/`:

- `rewardlab.sqlite3` for metadata
- `events.jsonl` for append-only event logs
- `checkpoints/*.json` for interruption/resume snapshots
- `reports/*.json` for exported session reports

Feedback-enabled sessions also persist reviewer entries in SQLite and thread
demo artifact references plus gate-ready recommendation summaries through
exported session reports.

## Engineering Standards

Standards are defined in `.specify/memory/constitution.md`, including modularity,
required module headers, required routine docstrings, and dead-code cleanup
before merge.
